"""Managed BUNDLE storage: staged whole-directory commits and journal recovery.

WHY A SECOND STORAGE CLASS. ``ManagedSkillStorage`` stages text files built from
a ``SkillDocument``; ``from_directory`` refuses every non-text byte, so that
class can never hold a published bundle. A bundle installs the archive's decoded
TREE, and its content identity is the reproducible ``tree_hash`` (never the
archive hash, which depends on the deflate implementation). The journal
discipline, alias ownership, policy materialization and baseline finalization
are deliberately the SAME as the document path -- a second, weaker commit path
is the failure this module exists to prevent -- so the differences are kept to
exactly three: the payload marker ``managed-skill-bundle``, the staging
location, and the fingerprint function.

STAGING INSIDE THE MANAGED ROOT. ``<config>/skills/.stage-<op>/content`` is
required by the operator and guarantees the final ``os.replace`` stays on one
filesystem. Skill discovery skips dot-prefixed directories
(``local_operator/skills/discovery.py``), so a live stage is invisible to every
reader; aliases can never collide with it (``validate_name`` is kebab-case).

NO MODE FROM THE ARCHIVE, EVER. Files are created 0600 and directories 0700 by
``skill_bundle.extract_archive``; this module never chmods and never runs a
hook. The appended ``+x`` on a script in the source directory is not published,
which is the intended consequence of the no-executables policy (design §6.1).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

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
from local_operator.hub_sync.skill_bundle import (
    SkillBundleManifest,
    bundle_manifest_from_directory,
    extract_archive,
    tree_hash_of,
)
from local_operator.hub_sync.skill_document import validate_name
from local_operator.hub_sync.transactions import (
    LocalTransaction,
    TransactionError,
    TransactionJournal,
    _sync_ancestors,
    _sync_directory,
)

logger = logging.getLogger(__name__)

_STAGE_NAME = re.compile(r"^\.stage-([a-f0-9]{32})$")
_OPERATION = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class BundleSnapshot:
    """One bundle link's local reading: its manifest and reproducible identity."""

    local_id: str
    alias: str
    manifest: SkillBundleManifest
    fingerprint: str


class ManagedBundleStorage:
    """Whole-directory commits of an installed bundle, journalled like text pulls."""

    #: The payload marker that keeps the two adapters' journals apart. Recovery
    #: reads this instead of a kind: one skill journal carries both forms, and
    #: each recovery pass must skip transactions it cannot settle.
    ADAPTER = "managed-skill-bundle"

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

    def _stage_root(self, operation_id: str) -> Path:
        if not _OPERATION.fullmatch(operation_id):
            raise LinkError("skill-validation", "bundle operation id is invalid")
        return self.root / f".stage-{operation_id}"

    # -- reads ---------------------------------------------------------------------

    def snapshot(self, local_id: str) -> BundleSnapshot:
        with config_write_lock(self.config_dir):
            self.recover(local_id=local_id)
            record = read_baseline(self.config_dir, "skill", local_id)
            if record is None or record.local_alias is None:
                raise LinkError("baseline-unknown", "managed skill has no recorded local alias")
            if "bundle" not in record.fields:
                raise LinkError(
                    "bundle-form-changed", "the recorded baseline is not a published bundle"
                )
            if record.content_state == "absent":
                raise LinkError("local-concurrent-edit", "managed skill is absent locally")
            destination = self.path(record.local_alias)
            if not destination.exists():
                raise LinkError("local-concurrent-edit", "managed skill is missing locally")
            if not destination.is_dir():
                raise LinkError(
                    "alias-occupied",
                    f"a file named {record.local_alias!r} is already in the skills folder; "
                    "move it somewhere else, then try again",
                )
            # The local reading must reflect ACTUAL bytes: a locally edited tree
            # has a different tree hash from the baseline, which is exactly what
            # ``classify`` needs to report a local change.
            manifest = bundle_manifest_from_directory(destination)
            return BundleSnapshot(local_id, record.local_alias, manifest, manifest.fingerprint)

    # -- writes --------------------------------------------------------------------

    def _stage(
        self, transaction: LocalTransaction, archive: Path, manifest: SkillBundleManifest
    ) -> Path:
        stage = self._stage_root(transaction.operation_id)
        if self.root.is_symlink():
            raise LinkError("skill-validation", "managed skill root is unsafe")
        self.root.mkdir(mode=0o700, exist_ok=True)
        if stage.is_symlink():
            raise LinkError("skill-validation", "managed bundle staging is unsafe")
        try:
            os.mkdir(stage, 0o700)
        except FileExistsError:
            raise LinkError(
                "local-concurrent-edit", "managed bundle staging already exists"
            ) from None
        content = stage / "content"
        try:
            # Validate + unpack in ONE pass into the stage; every failure removes
            # the stage so no half-written tree can be renamed into place.
            extract_archive(
                archive,
                content,
                expected=manifest.bundle,
                expected_name=manifest.name,
                expected_description=manifest.description,
            )
        except BaseException as exc:
            shutil.rmtree(stage, ignore_errors=True)
            # Settle the journal now: the caller's exception is the diagnostic,
            # but recovery must not later promote a tree this call refused.
            self.journal.transition(
                transaction, "aborted", error_class=str(getattr(exc, "code", ""))
            )
            raise
        return content

    def _locations(self, transaction: LocalTransaction) -> tuple[Path, Path, Path]:
        # validate_name AUTHENTICATES the payload value and returns it; binding
        # the return carries that runtime verification into the type here.
        alias = validate_name(transaction.payload.get("local_alias"))
        destination = self.path(alias)
        staged = self._stage_root(transaction.operation_id) / "content"
        backups = self.config_dir / "hub" / "backups"
        backup = backups / f"skill-{transaction.local_id}-{transaction.operation_id}"
        for directory in (
            self.config_dir / "hub",
            backups,
            backup,
            self.root,
            staged.parent,
            staged,
        ):
            if directory.is_symlink():
                raise LinkError("skill-validation", "managed skill transaction path is unsafe")
        return destination, staged, backup

    def _fingerprint_path(self, path: Path) -> str | None:
        """The content identity of a directory: its tree hash, or None if absent."""

        if not path.exists():
            return None
        if path.is_symlink() or not path.is_dir():
            raise LinkError("local-concurrent-edit", "managed skill content is not a directory")
        return tree_hash_of(path)

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
        # Parse the baseline through the BUNDLE manifest, so the recorded
        # fingerprint must equal the tree hash its fields declare. A document
        # parsed here would fail the form check instead of silently accepting.
        if SkillBundleManifest.from_json(dict(restored.fields)).fingerprint != restored.fingerprint:
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
        archive: Path,
        manifest: SkillBundleManifest,
        baseline: BaselineRecord,
        *,
        expected_fingerprint: str | None,
        expected_policy_revision: str | None = None,
        automatic: bool = False,
        basis: str = "integrated-baseline",
        checkpoint: Callable[[str], None] | None = None,
    ) -> LocalTransaction:
        """Install one already-downloaded archive as the alias's whole content.

        The archive is the ONLY input: its decoded tree replaces the destination
        atomically after the same preconditions a text pull re-checks under the
        writer lock (alias ownership, destination hash, policy revision). The
        caller has already verified the manifest against the hub; this method
        re-verifies the bytes against the manifest because a local writer could
        have touched the download in between.
        """

        if identity.kind != "skill" or baseline.identity != identity:
            raise LinkError("origin-conflict", "managed skill integration identity does not match")
        validate_name(alias)
        # A bundle's SKILL.md IS its bytes; renaming the alias would require
        # rewriting the entrypoint and breaks tree-hash equality (§6.2). This is
        # checked BEFORE the baseline/alias agreement so a rename attempt gets
        # the bundle-specific refusal rather than a generic origin conflict.
        if manifest.name != alias:
            raise LinkError(
                "bundle-alias-rename-unsupported",
                "a bundle is installed under its published name; "
                "renaming would rewrite its SKILL.md",
            )
        if baseline.local_alias != alias:
            raise LinkError("origin-conflict", "managed skill integration identity does not match")
        # Test fault-injection ONLY; production notifications run after return,
        # never inside the config/definition transaction lock.
        notify = checkpoint or (lambda _phase: None)
        with config_write_lock(self.config_dir):
            self.recover(local_id=identity.local_id)
            current_record = read_baseline(self.config_dir, "skill", identity.local_id)
            current = self._fingerprint_path(self.path(alias))
            # ALIAS OWNERSHIP IS DURABLE, NOT LIVE-DIRECTORY-LOCAL: a crash
            # between the two renames leaves the alias absent while its baseline
            # still owns it, so the durable record -- not the live directory --
            # decides whether another origin may take the name.
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
                raise LinkError(
                    "alias-occupied",
                    f"a skill named {alias!r} already exists on this machine outside the hub; "
                    "move it elsewhere or give it a different name, then try again",
                )
            before = current
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
                after_fingerprint=manifest.fingerprint,
                payload={
                    "adapter": self.ADAPTER,
                    "local_alias": alias,
                    "baseline": baseline.to_json(),
                    "policy_revision": policy.revision,
                    "automatic": automatic,
                    # The archive identity is transport evidence, kept beside the
                    # journal's tree fingerprints so recovery can name what was
                    # attempted without re-reading the download.
                    "archive_hash": manifest.bundle.hash,
                    "before_tree": before,
                    "after_tree": manifest.fingerprint,
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
            # Slow hashing/fsync preparation is outside the short read/commit
            # lock. Actual bytes/policy are checked again when the stage is ready.
            self._stage(transaction, Path(archive), manifest)
            notify("staged")
            with config_write_lock(self.config_dir):
                destination, staged, backup = self._locations(transaction)
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
                if self._fingerprint_path(staged) != manifest.fingerprint:
                    self.journal.transition(
                        transaction, "blocked", error_class="local-concurrent-edit"
                    )
                    raise LinkError(
                        "local-concurrent-edit", "managed staging changed before commit"
                    )
                self.root.mkdir(mode=0o700, exist_ok=True)
                backup.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                # A rename is durable only in the directory whose entries
                # changed: the backup's newly-created ancestors and the skills
                # root are fsynced once, before the swap (never a blanket fsync
                # of unrelated tree members -- reviewer R4).
                _sync_ancestors(backup.parent, self.config_dir / "hub")
                _sync_directory(self.root)
                if before != manifest.fingerprint:
                    if destination.exists():
                        os.replace(destination, backup)
                    os.replace(staged, destination)
                    _sync_directory(self.root)
                    notify("new-renamed")
                if self._fingerprint_path(destination) != manifest.fingerprint:
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
            # A SETTLED transaction needs no staged evidence. Anything else
            # (prepared/blocked/content-committed) is left for ``recover`` --
            # deleting a stage mid-swap would strand the backup it must restore.
            self._discard_settled_stage(transaction)

    def _discard_settled_stage(self, transaction: LocalTransaction) -> None:
        try:
            record = self.journal.read(transaction.operation_id)
        except TransactionError:
            return
        if record is not None and record.phase not in ("complete", "aborted"):
            return
        shutil.rmtree(self._stage_root(transaction.operation_id), ignore_errors=True)

    def recover(self, *, local_id: str | None = None) -> None:
        """Settle crashed bundle commits; never touch a document transaction."""

        with config_write_lock(self.config_dir):
            for transaction in self.journal.pending(kind="skill", local_id=local_id):
                if transaction.payload.get("adapter") != self.ADAPTER:
                    continue
                with self.journal.claim(transaction.operation_id) as claimed:
                    if claimed:
                        self._recover_one(transaction)
            self._cleanup_stages()

    def _cleanup_stages(self) -> None:
        """Remove every stage whose journal entry is settled or absent.

        Only our own ``.stage-<32-hex>`` names are candidates (never a user's
        dot-directory), and a live/prepared/blocked transaction keeps its stage
        because recovery may still need it.
        """

        if not self.root.is_dir() or self.root.is_symlink():
            return
        try:
            children = list(self.root.iterdir())
        except OSError:
            return
        for child in children:
            match = _STAGE_NAME.fullmatch(child.name)
            if match is None or child.is_symlink() or not child.is_dir():
                continue
            try:
                record = self.journal.read(match.group(1))
            except TransactionError:
                continue
            if record is None or record.phase in ("complete", "aborted"):
                shutil.rmtree(child, ignore_errors=True)

    def _recover_one(self, transaction: LocalTransaction) -> None:
        if transaction.phase == "blocked":
            return
        # The payload copies are written once with the record; a mismatch means a
        # hand-edited journal, which must never choose the restore target.
        if (
            transaction.payload.get("before_tree", transaction.before_fingerprint)
            != transaction.before_fingerprint
            or transaction.payload.get("after_tree", transaction.after_fingerprint)
            != transaction.after_fingerprint
        ):
            self.journal.transition(transaction, "blocked", error_class="local-concurrent-edit")
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
            # A crash in the two-rename gap restores the real original; it does
            # not blindly finish a now-unconfirmed prepared pull.
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

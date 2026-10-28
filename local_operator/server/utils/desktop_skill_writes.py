"""Skill-library WRITES for ``POST|PUT|DELETE /v1/desktop/skills`` (``skill_library:1``).

WHY THIS IS A SEPARATE MODULE from ``desktop_skills`` (the read projections): that
one is a pure projection and stays write-free by design. Everything here mutates
``<config_dir>/skills`` and nothing else, and every rule the Skills tab relies on
for safety lives in one place so the route handlers stay translation of HTTP shapes.

THE ONE WRITE DESTINATION. ``<config_dir>/skills/<name>`` is the managed root and
the only place a write may land. ``builtin``, ``external`` (ecosystem) and
``project`` rows are READ-ONLY: a write that names one is ``skill-not-editable``,
never a quiet write into the managed root (which would shadow the original and
read as an edit of it). "Copy to my skills" is the sanctioned way to change one:
``create`` with ``copy_from``.

A LOCAL EDIT IS NOT AN INTEGRATION. Every create/update goes through
``ManagedSkillStorage.author``, which leaves a linked skill's baseline and policy
exactly where they were: the gap between baseline and folder IS the "local
change" the link check reports as ``local-only``/``both-changed``. See the storage
docstring; routing an edit through ``apply`` would erase that signal.

SAFETY comes from the validators the hub already trusts rather than a second set:
``validate_name`` / ``validate_path`` / ``SkillDocument`` (no absolute, ``..``,
dot-leading, ``\\``, ``:`` or invisible-Unicode paths; only ``SKILL.md`` or files
under ``references/``, ``scripts/``, ``assets/``; 200 KiB per file and 1 MiB per
skill locally), ``SkillDocument.from_directory`` (refuses symlinks and binaries in
what is already on disk) and ``ManagedSkillStorage.path`` (refuses a symlinked
skills root or skill directory). The HUB's smaller limits are never enforced on a
local save -- a local save must not be blocked by a publication cap -- the caller
is told the budget instead (``publish_budget``) so the editor can warn.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from local_operator.config_writer import config_write_lock
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import LinkError
from local_operator.hub_sync.skill_document import (
    SkillDocument,
    SkillFile,
    SkillValidationError,
    parse_frontmatter,
    validate_name,
)
from local_operator.hub_sync.skill_storage import AuthoredSkill, ManagedSkillStorage
from local_operator.server.utils import desktop_skills

#: What a managed folder's owner is, for the caller: no link, or a hub link.
LOCAL = "local"
HUB = "hub"


def document_from_files(name: str, files: Iterable[Mapping[str, str]]) -> SkillDocument:
    """The validated document a request describes.

    The description is NOT a request field: it lives in ``SKILL.md``'s frontmatter,
    and the document refuses a manifest whose two copies disagree, so deriving it
    here removes the only way for them to. Every path/size/name rule runs in the
    ``SkillDocument`` constructor, so a hostile path is refused with the hub's own
    vocabulary before anything touches the disk.
    """
    validate_name(name)
    built = tuple(SkillFile(item["path"], item["content"]) for item in files)
    skill_md = next((file for file in built if file.path == "SKILL.md"), None)
    if skill_md is None:
        raise SkillValidationError("skill-validation", "skill requires root SKILL.md")
    meta = parse_frontmatter(skill_md.content)
    return SkillDocument(name, meta["description"], built)


def _discovered(config_dir: Path, name: str) -> Any | None:
    """The skill discovery resolves ``name`` to (any root), or ``None``.

    The same root set the read route uses for the home arm, so "read-only here"
    agrees with what the tab labels ``builtin``/``external``/``project``.
    """
    from local_operator.skills import default_skill_roots, discover_skills

    roots = default_skill_roots(Path.home())
    skills, _warnings = discover_skills(roots, config_dir=config_dir)
    return next((skill for skill in skills if skill.name == name), None)


def require_managed(config_dir: Path, name: str) -> ManagedSkillStorage:
    """Storage for ``name``, after proving the name is a skill THIS profile owns.

    * a name outside ``validate_name`` is a validation refusal (``.archive`` and
      every dot name fall out here);
    * a managed folder (linked or not) is the editable case;
    * a name that exists only in a read-only root, or whose link has no local copy,
      is ``skill-not-editable``;
    * a name that exists nowhere is ``skill-not-found``.
    """
    validate_name(name)
    storage = ManagedSkillStorage(config_dir)
    # Settle a crashed swap/unpull BEFORE judging what exists. After a crash
    # between the two renames the alias is absent from disk while its baseline
    # still owns it; judging first answered the user's first request with a
    # misleading "not editable" even though the very next step restores the skill.
    storage.recover()
    storage.recover_unpull()
    # ``path`` refuses a symlinked skills root or alias directory: a link is never
    # followed to bytes outside the store.
    destination = storage.path(name)
    if destination.exists():
        return storage
    if prov.alias_owner(config_dir, "skill", name) is not None:
        raise LinkError("skill-not-editable", "this skill has no local copy to change")
    if _discovered(config_dir, name) is not None:
        raise LinkError("skill-not-editable", "this skill is read-only here")
    raise LinkError("skill-not-found", "no such skill")


def _link_for(config_dir: Path, name: str) -> desktop_skills.SkillLink | None:
    return desktop_skills.link_index(config_dir).get(name)


def _result(config_dir: Path, name: str, authored: AuthoredSkill) -> dict[str, Any]:
    link = _link_for(config_dir, name)
    return {
        "name": name,
        "source": HUB if authored.linked else LOCAL,
        "link_id": link.link_id if link is not None else None,
        "form": "document",
        "editable": True,
        "fingerprint": authored.fingerprint,
        "previous_fingerprint": authored.before_fingerprint,
        # The link baseline's fingerprint AFTER the write. Equal to what it was
        # before BY CONSTRUCTION (author never touches it); returned so a caller
        # can show "differs from the published version" and a test can prove it.
        "baseline_fingerprint": authored.baseline_fingerprint,
        "changed": authored.changed,
        "publish_budget": desktop_skills.publish_budget(link, "document"),
    }


def create(
    config_dir: Path,
    name: str,
    *,
    files: Iterable[Mapping[str, str]] | None = None,
    copy_from: str | None = None,
) -> dict[str, Any]:
    """Create ``<config_dir>/skills/<name>``; an existing name is ``alias-occupied``."""
    validate_name(name)
    mode_source: Path | None = None
    if copy_from is not None:
        validate_name(copy_from)
        source = _discovered(config_dir, copy_from)
        if source is None:
            raise LinkError("skill-not-found", "no such skill to copy")
        # Resolve only the ROOT (a harness root may be a symlink into the store);
        # symlinks INSIDE the skill are still refused by ``from_directory``.
        mode_source = Path(os.path.realpath(source.base_dir))
        document = SkillDocument.from_directory(mode_source)
        if document.name != name:
            document = document.with_name(name)
    else:
        document = document_from_files(name, files or ())
    storage = ManagedSkillStorage(config_dir)
    authored = storage.author(name, document, None, mode_source=mode_source)
    return _result(config_dir, name, authored)


def update(
    config_dir: Path, name: str, files: Iterable[Mapping[str, str]], expected_fingerprint: str
) -> dict[str, Any]:
    """Replace the whole file set of an existing managed skill (CAS on the folder)."""
    storage = require_managed(config_dir, name)
    document = document_from_files(name, files)
    authored = storage.author(name, document, expected_fingerprint)
    return _result(config_dir, name, authored)


def folder_differs_from_baseline(storage: ManagedSkillStorage, record: prov.BaselineRecord) -> bool:
    """Whether the linked folder holds bytes the hub has never been given.

    Compared in the SOURCE-NAME space the link check uses (the local alias may
    differ from the remote name; renaming the local copy back before hashing is
    what keeps an alias from reading as a permanent edit). Anything unreadable --
    a symlink, a binary, a missing folder -- counts as "differs": the question
    guards a destructive step, so doubt must refuse it.
    """
    from local_operator.hub_sync.skill_bundle import tree_hash_of

    baseline = prov.fingerprint_fields("skill", record.fields)
    alias = record.local_alias or record.local_id
    try:
        path = storage.path(alias)
        if "bundle" in (record.fields or {}):
            return tree_hash_of(path) != baseline
        local = SkillDocument.from_directory(path)
        remote_name = str(record.fields.get("name") or "")
        if remote_name and local.name != remote_name:
            local = local.with_name(remote_name)
        return local.fingerprint != baseline
    except Exception:  # noqa: BLE001 - unreadable == cannot prove clean
        return True


def remove_local(config_dir: Path, name: str) -> dict[str, Any]:
    """Remove an UNLINKED managed skill by MOVING it to a kept backup (never a delete)."""
    require_managed(config_dir, name)
    backup = ManagedSkillStorage(config_dir).remove_local(name)
    return {
        "name": name,
        "removed": True,
        "mode": "backup",
        # Relative to the profile: the caller shows "kept at ...", and an absolute
        # path would put the operator's home directory into every log of the reply.
        "backup": backup.relative_to(Path(config_dir).expanduser().resolve()).as_posix(),
        "link_id": None,
    }


def remove_linked(
    ctx: Any, record: prov.BaselineRecord, *, discard_local_changes: bool
) -> dict[str, Any]:
    """Unpull a hub-linked skill (the link survives; only the local copy goes).

    ``unpull`` deletes its own tombstone once the state write is durable, so a
    folder that differs from the baseline would be lost for good. Hence: refused
    unless the caller said, in so many words, to discard -- and even then the
    discarded folder is first COPIED to ``hub/backups/skill-local-<name>-<op>``
    (the scheme a local remove uses), so a deliberate discard stays recoverable.

    THE LOCKS. The hub lease orders this against the runner; ``author`` does NOT
    take it, only ``config_write_lock``. So the dirty check and the unpull must
    also share ONE hold of ``config_write_lock``, or an edit can commit between
    "the folder is clean" and the removal and be destroyed unseen (review M1).
    The lock is re-entrant, so ``recover``/``unpull`` below simply nest in it. An
    edit already staging (live lease, not yet swapped) is refused instead of raced.
    """
    from local_operator.hub_sync import service as svc

    config_dir = ctx.config_dir
    storage = ManagedSkillStorage(config_dir)
    backup: Path | None = None
    with svc.exclusive(ctx, wait_s=0.0), config_write_lock(config_dir):
        # Settle crashes first so the comparison reads the bytes a reader would
        # see, not a half-swapped directory.
        storage.recover(local_id=record.local_id)
        storage.recover_unpull(local_id=record.local_id)
        current = prov.read_baseline(config_dir, "skill", record.local_id)
        if current is None or current.identity is None:
            raise LinkError("baseline-unknown", "the skill's link baseline is unreadable")
        if current.content_state != "absent":
            storage.refuse_live_transaction(record.local_id)
            if folder_differs_from_baseline(storage, current):
                if not discard_local_changes:
                    raise LinkError(
                        "skill-local-changes-unpublished",
                        "this skill has local changes that were never published",
                    )
                backup = storage.keep_copy(current.local_alias or current.local_id)
        storage.unpull(record.local_id)  # idempotent when already absent
    return {
        "name": record.local_alias or record.local_id,
        "removed": True,
        "mode": "unpull",
        "backup": (
            backup.relative_to(Path(config_dir).expanduser().resolve()).as_posix()
            if backup is not None
            else None
        ),
        "link_id": current.identity.link_id,
    }

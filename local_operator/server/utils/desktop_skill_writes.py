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


def run_skill_roots(config_dir: Path) -> list[Path]:
    """The discovery roots a CONFIGURATION RUN judges skills by (``tools/skill_tool``).

    ``_discovered``'s home-derived set, plus THIS profile's managed root when that
    walk does not reach it. With ``LOCAL_OPERATOR_CONFIG_DIR`` moved,
    ``<home>/.local-operator/skills`` is another profile's store and
    ``<config_dir>/skills`` is scanned by nothing, so a run could ``update`` a skill
    it could not ``show`` (and so never learn its fingerprint). It is inserted
    where the home native root sits -- after the project roots, ahead of the
    ecosystem and packaged ones -- the precedence it has in the default layout.

    ONE helper for both of the run's readers (its ``list``/``show`` and
    :func:`require_unshadowed`), so what the run is shown as editable and what its
    ``create`` refuses are decided over the same roots and answer with the same
    codes (review R3). The Skills tab's routes keep ``_discovered``: changing what
    they resolve under a moved profile is not this feature's to decide.
    """
    from local_operator.skills import default_skill_roots

    roots = default_skill_roots(Path.home())
    managed = Path(config_dir) / "skills"
    if os.path.realpath(managed) not in {os.path.realpath(root) for root in roots}:
        home_store = os.path.realpath(Path.home() / ".local-operator" / "skills")
        at = next(
            (i for i, root in enumerate(roots) if os.path.realpath(root) == home_store),
            len(roots) - 1,
        )
        roots.insert(at, managed)
    return roots


def require_unshadowed(config_dir: Path, name: str) -> None:
    """Refuse a CREATE whose name a read-only root already supplies.

    WHY THE ROUTE DOES NOT CALL THIS. ``create`` over a builtin/external/project
    name is exactly what "Copy to my skills" is: the operator, on the Skills tab,
    deliberately makes an editable copy that shadows the original. A configuration
    run (``tools/skill_tool.py``) is a model acting on a typed request, and a
    same-named local skill silently replaces the original for EVERY later session
    -- the same harm ``profile_scope.shadow_refusal`` exists to stop for agents.
    So the run must not, and the shadowing decision stays a click the operator
    makes.

    Only the read-only roots are judged here. A name the managed root (or a hub
    link) already owns is left to ``ManagedSkillStorage.author``, whose
    ``alias-occupied`` answer is the precise one for that case. The refusal is
    ``skill-not-editable`` because its remedy is the one that code already names
    -- the operator's own Copy to my skills -- not "pick another name".
    """
    from local_operator.skills import discover_skills

    validate_name(name)
    skills, _warnings = discover_skills(run_skill_roots(config_dir), config_dir=config_dir)
    found = next((skill for skill in skills if skill.name == name), None)
    if found is None:
        return
    managed = os.path.realpath(Path(config_dir) / "skills")
    base = os.path.realpath(found.base_dir)
    if base == managed or base.startswith(managed.rstrip(os.sep) + os.sep):
        return
    raise LinkError(
        "skill-not-editable",
        f"a read-only skill named {name!r} already exists; a same-named copy would "
        "replace it in every session",
    )


def _current_document(
    config_dir: Path, name: str, expected_fingerprint: str
) -> tuple[ManagedSkillStorage, SkillDocument]:
    """The managed skill's current document, after proving it is the one the caller read.

    Shared by :func:`patch` and :func:`edit_files`. Strict on purpose: unlike the
    route this has no lost-response retry to tolerate, so a stale fingerprint is
    always refused, even for an edit that would change nothing.
    """
    storage = require_managed(config_dir, name)
    # The same refusal ``author`` gives, made BEFORE the read: a bundle link may
    # hold binary files, and reading it as a text document would answer
    # "invalid skill" instead of the true "not editable here".
    owner = prov.alias_owner(config_dir, "skill", name)
    record = prov.read_baseline(config_dir, "skill", owner) if owner is not None else None
    if record is not None and (
        "bundle" in (record.fields or {}) or record.content_state == "absent"
    ):
        raise LinkError("skill-not-editable", "this skill's link has no editable text copy here")
    # Under the root lock: the swap holds it, so the read never sees a half-swapped
    # directory.
    with config_write_lock(config_dir):
        destination = storage.path(name)
        if not destination.is_dir():
            raise LinkError("skill-not-found", "no such skill in the managed skills folder")
        current = SkillDocument.from_directory(destination)
    if current.fingerprint != expected_fingerprint:
        raise LinkError("local-concurrent-edit", "the skill changed since it was read")
    return storage, current


def _commit(
    config_dir: Path,
    storage: ManagedSkillStorage,
    name: str,
    merged: Mapping[str, str],
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Author the overlaid file set; ``author`` re-checks the CAS under the lock."""
    document = document_from_files(
        name, ({"path": path, "content": content} for path, content in merged.items())
    )
    authored = storage.author(name, document, expected_fingerprint)
    return _result(config_dir, name, authored)


def _refuse_duplicates(paths: list[str], what: str) -> None:
    # A dict built from a list keeps the LAST of two same-path entries, so two
    # conflicting versions of one file would silently lose one (review R4).
    # ``create`` refuses the same input through SkillDocument's collision check.
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            raise SkillValidationError("skill-validation", f"{what} names {path} twice")
        seen.add(path)


def patch(
    config_dir: Path,
    name: str,
    files: Iterable[Mapping[str, str]],
    remove_paths: Iterable[str],
    expected_fingerprint: str,
    *,
    whole_file_max_chars: int | None = None,
) -> dict[str, Any]:
    """Change SOME files of a managed skill and keep the rest (CAS on the folder).

    WHY AN OVERLAY AND NOT ``update``. ``author`` replaces the whole file set, so
    an editor that sends only ``SKILL.md`` deletes every reference file. The
    Skills tab's editor sends the full set on purpose; a model revising one file
    will not reliably resend the others, and the loss would be silent. Here the
    named ``files`` replace or add, ``remove_paths`` delete, and every other file
    is carried over byte for byte.

    ``whole_file_max_chars`` refuses REPLACING an existing file longer than that
    (new files are unaffected). The configuration run passes the largest file one
    of its reads can deliver whole through the loop's replay bound: a longer file
    reached the model only in pages, and a whole-file rewrite composed from a
    partial view commits whatever the model filled the gaps with (review R1).
    :func:`edit_files` is the path for those.

    THE CAS IS CHECKED TWICE. Once against the document the overlay is built on --
    an overlay on bytes the caller never saw would merge their edit into someone
    else's -- and again by ``author`` under ``config_write_lock`` right before the
    swap, which is what closes the gap between this read and the write.
    """
    files = list(files)
    removed = list(remove_paths)
    _refuse_duplicates([item["path"] for item in files], "files")
    _refuse_duplicates(removed, "remove_paths")
    storage, current = _current_document(config_dir, name, expected_fingerprint)
    merged = {file.path: file.content for file in current.files}
    changed = {item["path"]: item["content"] for item in files}
    both = sorted(set(changed) & set(removed))
    if both:
        raise SkillValidationError(
            "skill-validation", f"a path cannot be both written and removed: {both[0]}"
        )
    if whole_file_max_chars is not None:
        for path in changed:
            existing = merged.get(path)
            if existing is not None and len(existing) > whole_file_max_chars:
                raise SkillValidationError(
                    "skill-file-too-long-to-replace",
                    f"{path} is {len(existing)} characters, more than one read shows "
                    f"whole ({whole_file_max_chars}), so it cannot be replaced whole",
                )
    for path in removed:
        if path not in merged:
            raise SkillValidationError("skill-validation", f"no file to remove: {path}")
        del merged[path]
    merged.update(changed)
    return _commit(config_dir, storage, name, merged, expected_fingerprint)


def edit_files(
    config_dir: Path,
    name: str,
    edits: Iterable[Mapping[str, str]],
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Exact ``old_text`` -> ``new_text`` replacements in a managed skill's files.

    WHY THIS EXISTS. A configuration run reads files through a replay bound, so a
    long file reaches the model in pages and a whole-file rewrite of it is refused
    (:func:`patch`). An anchored replacement needs only the text it replaces, so
    the model can change any part it has read and every byte it did not touch is
    carried over unchanged -- the property is that no write is built from content
    the model never received.

    The semantics are the file ``edit`` tool's, minus its whitespace-tolerant
    fallback (a skill is short and the CAS already proves the model read this
    revision): ``old_text`` must be non-empty and occur EXACTLY ONCE in the file
    as it stands after the previous edits in this call, or nothing is written.
    All edits land in one swap.
    """
    edits = list(edits)
    storage, current = _current_document(config_dir, name, expected_fingerprint)
    merged = {file.path: file.content for file in current.files}
    for index, item in enumerate(edits, start=1):
        path, old, new = item["path"], item["old_text"], item["new_text"]
        if path not in merged:
            raise SkillValidationError("skill-validation", f"edit {index}: no file {path}")
        if not old:
            raise SkillValidationError("skill-validation", f"edit {index}: old_text is empty")
        count = merged[path].count(old)
        if count != 1:
            raise SkillValidationError(
                "skill-edit-no-unique-match",
                f"edit {index}: old_text occurs {count} times in {path}; it must occur "
                "exactly once (copy it exactly from show, with enough context to be unique)",
            )
        merged[path] = merged[path].replace(old, new, 1)
    return _commit(config_dir, storage, name, merged, expected_fingerprint)


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

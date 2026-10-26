"""Read-only skill-library projections for ``GET /v1/desktop/skills``.

WHY THIS EXISTS. The Agents page Skills tab draws cards and a details view, and
until now the route returned only ``{name, description}`` and threw away the
one fact a card needs most: WHERE the skill lives, which decides whether it can
be edited, copied, published or only read. Everything here is a projection over
facts discovery and the hub link store already hold; nothing is written, and no
hub state (visibility, organisation, subscription, update-available) is copied
into the list. The UI joins ``link_id`` to the links inventory it already loads,
so there is exactly one owner of that state.

SOURCE LABELS, checked in this order on REAL paths (a symlinked root must be
labelled by where the bytes live, and ``LOCAL_OPERATOR_CONFIG_DIR`` can move the
managed store, so nothing here hard-codes ``~``):

1. ``builtin``  -- the skill's directory is inside the packaged catalogue.
2. ``hub``      -- discovered in ``<config_dir>/skills`` AND a link baseline
   records that alias.
3. ``local``    -- discovered in ``<config_dir>/skills`` with no link.
4. ``external`` -- discovered in an ecosystem root (``~/.omp``, ``~/.claude``,
   ``~/.codex``, ``~/.agents``, or the ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` set).
5. ``project``  -- any other root (a walk-up project folder's own skills root).

Discovery dedupes by realpath and scans the managed root BEFORE the ecosystem
roots, so a harness root that is a symlink into the managed store contributes
nothing and each skill appears once, labelled by its real home.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from local_operator.hub_sync.skill_document import (
    HUB_PUBLISH_BUDGETS,
    MAX_FILES,
    MAX_PATH_SEGMENTS,
    SkillDocument,
)

#: How much of each reference file a details read may return. The details view
#: shows what a file IS, not the file: full content stays behind the existing
#: ``skill://<name>/<path>`` resolver, which has its own caps.
PREVIEW_BYTES = 2048

#: Directory entries the file listing will look at before it stops. The listing
#: is advisory UI data, so a hostile or accidental huge tree is cut off rather
#: than walked (``files_truncated`` tells the caller).
_MAX_LISTED_ENTRIES = MAX_FILES * MAX_PATH_SEGMENTS


@dataclass(frozen=True)
class SkillLink:
    """The slice of a link baseline a skill row needs (never its policy)."""

    link_id: str | None
    visibility: str | None
    form: str
    content_present: bool
    description: str


def _real(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(path)


def _inside(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def link_index(config_dir: Path) -> dict[str, SkillLink]:
    """Skill link baselines by managed alias, from the same store the links route reads.

    The alias key is ``local_alias or local_id`` -- exactly how ``links_snapshot``
    names a row -- so a card and the links inventory agree on which skill a link
    belongs to. A link without a confirmed origin identity still marks its alias
    as hub-managed but carries no ``link_id`` (the inventory lists it under
    ``unresolved``).
    """
    from local_operator.hub_sync import provenance as prov

    index: dict[str, SkillLink] = {}
    for record in prov.iter_baselines(config_dir):
        if record.kind != "skill":
            continue
        identity = record.identity
        description = record.fields.get("description") if record.fields else None
        index[record.local_alias or record.local_id] = SkillLink(
            link_id=identity.link_id if identity is not None else None,
            visibility=identity.visibility if identity is not None else None,
            # Same discriminator ``links_snapshot`` uses for ``form``.
            form="bundle" if "bundle" in (record.fields or {}) else "document",
            content_present=record.content_state != "absent",
            description=description if isinstance(description, str) else "",
        )
    return index


class SourceClassifier:
    """Labels a discovered skill; built once per request (realpaths are not free)."""

    def __init__(self, config_dir: Path, ecosystem_roots: list[Path], links: Mapping[str, Any]):
        from local_operator.skills.discovery import packaged_skill_root

        self._builtin = _real(packaged_skill_root())
        self._managed = _real(Path(config_dir) / "skills")
        self._ecosystem = [_real(root) for root in ecosystem_roots]
        self._links = links

    def classify(self, skill: Any) -> str:
        base = _real(skill.base_dir)
        if _inside(base, self._builtin):
            return "builtin"
        # ``skill.source`` is the ROOT discovery reached the skill through. Its
        # realpath, not the base dir's, names the label: a managed skill that is
        # itself a symlink elsewhere is still the managed root's entry.
        root = _real(skill.source)
        if root == self._managed:
            return "hub" if skill.name in self._links else "local"
        if root in self._ecosystem:
            return "external"
        return "project"


def _fingerprint_and_form(base_dir: Path, linked_form: str | None) -> tuple[str | None, str | None]:
    """``(fingerprint, form)`` of the folder's bytes; ``(None, None)`` when unreadable.

    The fingerprint is the SAME function the managed storage compares against a
    link baseline (``SkillDocument.fingerprint`` for a text document, the tree
    hash for a bundle), so a card's fingerprint is comparable to the link's
    without a second hashing rule. A linked row keeps its recorded form; an
    unlinked folder is a ``document`` when it parses as one and a ``bundle``
    only when the tree hash accepts it (binary or oversize content), otherwise
    it has no form -- the honest answer is "not publishable as either".
    """
    from local_operator.hub_sync.skill_bundle import tree_hash_of

    def bundle() -> tuple[str | None, str | None]:
        try:
            return tree_hash_of(base_dir), "bundle"
        except Exception:  # noqa: BLE001 - a refusal means "no such form", not an error
            return None, None

    if linked_form == "bundle":
        fingerprint, _ = bundle()
        return fingerprint, "bundle"
    try:
        return SkillDocument.from_directory(base_dir).fingerprint, "document"
    except Exception:  # noqa: BLE001 - the folder is not a valid text document
        if linked_form == "document":
            return None, "document"
        return bundle()


def build_rows(
    discovered: list[Any], classifier: SourceClassifier, links: Mapping[str, SkillLink]
) -> list[dict[str, Any]]:
    """One additive row per discovered skill, then the subscribed-not-installed links.

    The first two keys keep the historical ``{name, description}`` contract; every
    other key is additive and optional, so an older caller sees a strict superset.
    """
    rows: list[dict[str, Any]] = []
    for skill in discovered:
        source = classifier.classify(skill)
        link = links.get(skill.name) if source == "hub" else None
        fingerprint, form = _fingerprint_and_form(
            Path(skill.base_dir), link.form if link is not None else None
        )
        rows.append(
            {
                "name": skill.name,
                "description": skill.description,
                "source": source,
                "link_id": link.link_id if link is not None else None,
                # Advisory in this read-only version (no write route exists yet):
                # what the future write path will accept -- the managed root,
                # text-document form only (bundle links are read-only in v1).
                "editable": source in ("hub", "local") and form == "document",
                "installed": True,
                "fingerprint": fingerprint,
                "form": form,
            }
        )
    present = {row["name"] for row in rows}
    for alias, link in sorted(links.items()):
        # A subscribed link with no files on disk has no discovered row. A name a
        # DIFFERENT root already supplies is skipped: two cards with one name
        # would break every name-keyed consumer, and the shadowing copy is the
        # one the runtime actually uses.
        if link.content_present or link.link_id is None or alias in present:
            continue
        rows.append(
            {
                "name": alias,
                "description": link.description,
                "source": "hub",
                "link_id": link.link_id,
                "editable": False,
                "installed": False,
                "fingerprint": None,
                "form": link.form,
            }
        )
    return rows


def _walk_files(base: Path) -> Iterator[tuple[str, Path]]:
    """Regular, non-dot, non-symlink files under ``base`` with their relative path.

    Mirrors the resolver's own listing rules (``protocol._reference_listing``):
    dotfiles and symlinks are never advertised, so the details view cannot list
    a path a follow-up ``skill://`` read would refuse. Depth and entry count are
    bounded because this walks user-controlled folders.
    """
    seen = 0
    pending: list[tuple[Path, int]] = [(base, 1)]
    while pending:
        directory, depth = pending.pop()
        try:
            with os.scandir(directory) as children:
                entries = sorted(children, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen > _MAX_LISTED_ENTRIES:
                return
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                if depth < MAX_PATH_SEGMENTS:
                    pending.append((path, depth + 1))
            elif entry.is_file(follow_symlinks=False):
                yield path.relative_to(base).as_posix(), path


def _preview(path: Path) -> tuple[str | None, bool]:
    """A bounded text preview, or ``(None, False)`` for non-UTF-8 content."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(PREVIEW_BYTES + 1)
    except OSError:
        return None, False
    truncated = len(raw) > PREVIEW_BYTES
    try:
        return raw[:PREVIEW_BYTES].decode("utf-8", errors="strict"), truncated
    except UnicodeError:
        # A multi-byte character cut by the bound is still text; only a body
        # that is not UTF-8 even before the cut is reported as binary.
        try:
            return raw[: PREVIEW_BYTES - 3].decode("utf-8", errors="strict"), True
        except UnicodeError:
            return None, False


def publish_budget(link: SkillLink | None, form: str | None) -> dict[str, int] | None:
    """The hub's caps for this link's scope, for a linked TEXT-document skill only.

    ``limit`` is the per-file cap and ``total`` the summed cap, in bytes
    (:data:`HUB_PUBLISH_BUDGETS`). Bundle links are whole-directory artifacts
    governed by other limits, and an unlinked skill has no destination yet.
    """
    if link is None or link.visibility is None or form != "document":
        return None
    caps = HUB_PUBLISH_BUDGETS.get(link.visibility)
    if caps is None:
        return None
    limit, total = caps
    return {"limit": limit, "total": total}


def build_detail(row: dict[str, Any], skill: Any | None, link: SkillLink | None) -> dict[str, Any]:
    """The details read: the row, the SKILL.md body, a bounded file list, the budget.

    ``skill`` is ``None`` for a subscribed-not-installed link: there are no
    bytes, so the body is ``None`` and ``files`` is empty rather than a 404 --
    the card exists and so does its details view.
    """
    from local_operator.skills.protocol import _read_text_capped

    body: str | None = None
    body_bytes: int | None = None
    files: list[dict[str, Any]] = []
    truncated = False
    if skill is not None:
        body = _read_text_capped(Path(skill.file_path))
        try:
            body_bytes = Path(skill.file_path).stat().st_size
        except OSError:
            body_bytes = None
        base = Path(skill.base_dir)
        listed = 0
        for relative, path in sorted(_walk_files(base)):
            if relative == "SKILL.md":
                continue
            if listed >= MAX_FILES:
                truncated = True
                break
            listed += 1
            try:
                size = path.stat().st_size
            except OSError:
                continue
            preview, preview_truncated = _preview(path)
            files.append(
                {
                    "path": relative,
                    "size": size,
                    "preview": preview,
                    "preview_truncated": preview_truncated,
                }
            )
    return {
        **row,
        "body": body,
        "body_bytes": body_bytes,
        "files": files,
        "files_truncated": truncated,
        "publish_budget": publish_budget(link, row.get("form")),
    }

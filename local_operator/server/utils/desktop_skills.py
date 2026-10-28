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
from typing import Any, Mapping

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
    """Labels a discovered skill; built once per request (realpaths are not free).

    THE FOLDER'S REAL PATH DECIDES builtin/hub/local, never the root discovery
    happened to reach it through. The two differ whenever the store is reached
    by a symlink: ``default_skill_roots`` is HOME-derived and never scans a
    moved ``<config_dir>/skills``, so with ``LOCAL_OPERATOR_CONFIG_DIR`` moved
    the managed skills are discovered ONLY through ``~/.claude|.codex|.agents``
    links into it (QA Q1). Labelling by the discovery root called those
    ``external`` with no link, and the identical layout at the default dir
    ``hub`` -- the label would then depend on where the profile lives.
    """

    def __init__(self, config_dir: Path, ecosystem_roots: list[Path], links: Mapping[str, Any]):
        from local_operator.skills.discovery import packaged_skill_root

        self._builtin = _real(packaged_skill_root())
        self._managed = _real(Path(config_dir) / "skills")
        # The GLOBAL native root discovery scans (``default_skill_roots`` puts
        # ``<home>/.local-operator/skills`` there). When the config dir is moved
        # it is another profile's store, not this one's: read-only here.
        self._home_store = _real(Path.home() / ".local-operator" / "skills")
        self._ecosystem = [_real(root) for root in ecosystem_roots]
        self._links = links

    def classify(self, skill: Any) -> str:
        base = _real(skill.base_dir)
        if _inside(base, self._builtin):
            return "builtin"
        if _inside(base, self._managed):
            return "hub" if skill.name in self._links else "local"
        # Not this profile's store. A skill under the default store while the
        # profile has moved elsewhere belongs to another profile, and one that
        # discovery reached through the managed root but that really lives
        # elsewhere (a symlinked entry) is not stored here either: both are
        # ``external`` (read-only), never ``project`` -- they are not a folder's
        # own skills.
        if _inside(base, self._home_store) or _real(skill.source) == self._managed:
            return "external"
        if _real(skill.source) in self._ecosystem:
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

    # The readers refuse a symlinked ROOT (that is how they refuse a skill whose
    # root was swapped under them) but a skill directory that is itself a link --
    # the migration's ``~/.claude/skills/<n> -> <store>/<n>`` -- is a legitimate
    # way to reach real bytes. Resolve only the root: symlinks INSIDE the skill
    # are still refused by the readers.
    base_dir = Path(_real(base_dir))

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


def _walk_files(base: Path) -> tuple[list[tuple[str, Path]], bool]:
    """Regular, non-dot, non-symlink files under ``base`` as ``(relative, path)``, plus a cut flag.

    Mirrors the resolver's own listing rules (``protocol._reference_listing``):
    dotfiles and symlinks are never advertised, so the details view cannot list
    a path a follow-up ``skill://`` read would refuse. Depth and entry count are
    bounded because this walks user-controlled folders; the flag says the entry
    bound stopped the walk, so a count taken from it is a lower bound.
    """
    found: list[tuple[str, Path]] = []
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
                return found, True
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                if depth < MAX_PATH_SEGMENTS:
                    pending.append((path, depth + 1))
            elif entry.is_file(follow_symlinks=False):
                found.append((path.relative_to(base).as_posix(), path))
    return found, False


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


def build_detail(
    row: dict[str, Any],
    skill: Any | None,
    link: SkillLink | None,
    resolver_text: str | None = None,
) -> dict[str, Any]:
    """The details read: the row, the SKILL.md body, a bounded file list, the budget.

    ``skill`` is ``None`` for a subscribed-not-installed link: there are no
    bytes, so the body is ``None`` and ``files`` is empty rather than a 404 --
    the card exists and so does its details view.

    ``link`` is honoured only for a ``hub`` row: the link index is keyed by
    alias, and a project/external copy that shadows the alias is not the linked
    skill, so it has no destination and no publish budget (review M1).

    THE BODY IS NOT SENT TWICE. The route's historical ``detail`` is the
    resolver's text, which is exactly ``body + "\n" + reference listing`` and
    must stay unchanged for released callers. When ``resolver_text`` starts with
    the body this read would return, ``body`` is ``None`` and ``body_in_detail``
    says to take ``detail[:body_length]`` instead (a 1 MiB SKILL.md was a 417 KB
    response, QA Q4). When it does not (a concurrent edit between the two reads),
    the body is inlined so the pair can never disagree.

    ``body_length`` is in UTF-16 CODE UNITS, i.e. a JavaScript string's
    ``length``, because its only consumer slices ``detail`` with it in JS. A
    Python ``len`` counts code points, which diverges from that for every
    astral-plane character (any emoji counts 1 there, 2 in JS) and would cut
    the body short; BMP-only text hides the difference, so a test must use an
    emoji. It measures the body text EXCLUDING the reader's truncation marker:
    ``body_truncated`` already says the body was cut, and the marker is the
    resolver's prose, not part of the file.
    """
    from local_operator.skills.protocol import (
        _TRUNCATION_MARKER,
        MAX_READ_BYTES,
        _read_text_capped,
    )

    if row.get("source") != "hub":
        link = None
    body: str | None = None
    body_length: int | None = None
    body_bytes: int | None = None
    body_truncated = False
    body_in_detail = False
    files: list[dict[str, Any]] = []
    files_total = 0
    walk_cut = False
    if skill is not None:
        text = _read_text_capped(Path(skill.file_path))
        try:
            body_bytes = Path(skill.file_path).stat().st_size
        except OSError:
            body_bytes = None
        # The reader cuts at ``MAX_READ_BYTES``; say so rather than make a
        # client compare ``body_bytes`` with the string length.
        body_truncated = body_bytes is not None and body_bytes > MAX_READ_BYTES
        if body_truncated and text.endswith(_TRUNCATION_MARKER):
            text = text[: -len(_TRUNCATION_MARKER)]
        # ``utf-16-le`` has no BOM; two bytes per unit, four per astral pair.
        body_length = len(text.encode("utf-16-le", errors="surrogatepass")) // 2
        # ``startswith`` is a code-point comparison on Python strings, so it is
        # independent of the UTF-16 unit count reported to the client.
        if resolver_text is not None and resolver_text.startswith(text):
            body_in_detail = True
        else:
            body = text
        listed, walk_cut = _walk_files(Path(skill.base_dir))
        candidates = sorted(item for item in listed if item[0] != "SKILL.md")
        files_total = len(candidates)
        for relative, path in candidates[:MAX_FILES]:
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
        "body_in_detail": body_in_detail,
        "body_length": body_length,
        "body_bytes": body_bytes,
        "body_truncated": body_truncated,
        "files": files,
        # ``files_total`` is a lower bound when the bounded walk itself was cut.
        "files_total": files_total,
        "files_truncated": walk_cut or files_total > len(files),
        "publish_budget": publish_budget(link, row.get("form")),
    }

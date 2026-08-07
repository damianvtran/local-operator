"""Skill discovery: scan ``<root>/<child>/SKILL.md`` directories into ``Skill`` records.

On-disk format is identical to the wider Agent Skills ecosystem:
a skill is a directory containing a ``SKILL.md`` whose YAML frontmatter
carries ``name``, ``description``, ``enabled``, ``hide`` and the Agent
Skills-standard ``disable-model-invocation``. The body is never read here —
it is fetched on demand through the ``skill://`` protocol, so discovery stays
cheap regardless of skill size.

Deliberate divergences from that ecosystem (see docs/REWRITE.md §C):

- Selection is semantic, done by :mod:`local_operator.skills.index`, so the
  description is the *routing signal* rather than guaranteed context.
- Roots are just the walk-up ``.local-operator/skills`` dirs plus the home
  root (see :func:`local_operator.skills.api.default_skill_roots`).

Invariants maintained for prompt-cache stability:

- Output order is deterministic: ``(name.lower(), name, file_path)``.
- Name collisions resolve to the EARLIEST root; losers are dropped with a
  warning rather than silently shadowed.
- The same physical ``SKILL.md`` is never loaded twice, even when two roots
  or symlinks point at it (realpath dedupe).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class Skill(BaseModel):
    """A discovered skill — everything the prompt and protocol layers need.

    ``base_dir`` is the skill directory (``file_path`` minus ``SKILL.md``);
    ``skill://<name>/<path>`` reads are joined against it with containment
    validation. ``source`` identifies the root the skill was scanned from,
    used only for warnings/diagnostics. ``hide`` keeps the skill readable
    via ``skill://`` while excluding it from prompt listings.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    file_path: Path
    base_dir: Path
    source: str
    hide: bool = False


def parse_frontmatter(text: str) -> dict[str, object]:
    """Parse the leading ``---`` YAML block of a SKILL.md.

    Returns ``{}`` when there is no frontmatter, the block is unterminated,
    or the YAML is malformed — a broken frontmatter must degrade to
    "missing description" (skill dropped) rather than crash startup.
    """
    if not text.startswith("---"):
        return {}
    lines = text.split("\n")
    end: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _skill_from_file(skill_md: Path, base_dir: Path, source: str) -> Skill | None:
    """Build one Skill from a SKILL.md, or None when it must be skipped.

    Drop rules: ``enabled: false`` skips entirely; a missing or
    blank description is dropped silently (the description is the whole
    routing signal, an unnamed one cannot be selected). ``hide`` is the OR of
    ``hide`` and ``disable-model-invocation``.
    """
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    meta = parse_frontmatter(text)

    # Authors write enabled: 0 or enabled: "false" expecting a disabled
    # skill; identity-comparing against the False singleton kept those
    # enabled. Any explicitly-present falsy value (and the string spellings)
    # disables.
    enabled = meta.get("enabled")
    if enabled is not None and (
        not enabled or str(enabled).strip().lower() in ("false", "no", "off")
    ):
        return None

    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        return None

    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        name = base_dir.name
    else:
        name = name.strip()

    hide = bool(meta.get("hide")) or bool(meta.get("disable-model-invocation"))
    return Skill(
        name=name,
        description=description.strip(),
        file_path=skill_md,
        base_dir=base_dir,
        source=source,
        hide=hide,
    )


def scan_skills_dir(
    dir: Path,
    source: str,
    include_self: bool = False,
    seen: set[str] | None = None,
) -> list[Skill]:
    """Scan one root non-recursively: ``<dir>/<child>/SKILL.md``.

    Dot-prefixed entries are skipped; symlinked child dirs are followed but
    deduped through ``seen`` (a set of realpaths shared across roots by
    :func:`discover_skills`) so one physical file is never loaded twice.
    ``include_self`` additionally accepts ``<dir>/SKILL.md`` (the wider
    ecosystem only uses this for Claude-plugin manifests; kept for parity).
    """
    skills: list[Skill] = []
    if seen is None:
        seen = set()

    def add(skill_md: Path, base_dir: Path) -> None:
        try:
            real = os.path.realpath(skill_md)
        except OSError:
            return
        if real in seen:
            return
        skill = _skill_from_file(skill_md, base_dir, source)
        if skill is None:
            return
        seen.add(real)
        skills.append(skill)

    if include_self:
        self_md = dir / "SKILL.md"
        if self_md.is_file():
            add(self_md, dir)

    try:
        children = sorted(dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return skills

    for child in children:
        if child.name.startswith("."):
            continue
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if skill_md.is_file():
            add(skill_md, child)

    return skills


def _sort_key(skill: Skill) -> tuple[str, str, str]:
    """Deterministic prompt order: case-insensitive name, exact name, path.

    Byte-stable ordering across turns keeps the volatile skills block from
    churning provider prompt caches (matching the established behavior).
    """
    return (skill.name.lower(), skill.name, str(skill.file_path))


def discover_skills(roots: Sequence[Path]) -> tuple[list[Skill], list[str]]:
    """Discover skills across roots; earlier roots win name collisions.

    Returns ``(skills, warnings)``. Warnings name every shadowed skill so a
    user who wonders why an edit to a home-root skill "did nothing" can see
    that a project-root skill of the same name took precedence. Missing
    roots are silently skipped (the home root may not exist yet).
    """
    seen: set[str] = set()
    warnings: list[str] = []
    ordered: list[Skill] = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        ordered.extend(scan_skills_dir(root, source=str(root), seen=seen))

    by_name: dict[str, Skill] = {}
    final: list[Skill] = []
    for skill in ordered:
        existing = by_name.get(skill.name)
        if existing is not None:
            warnings.append(
                f"Skill name conflict: '{skill.name}' from '{skill.file_path}' "
                f"shadowed by '{existing.name}' from '{existing.file_path}' "
                f"(earlier root wins)"
            )
            continue
        by_name[skill.name] = skill
        final.append(skill)

    final.sort(key=_sort_key)
    return final, warnings

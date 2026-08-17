"""Public surface of the skills subsystem (stream C).

Session and tools import ONLY from here — the lazy-import contract the
session relies on (a missing/broken skills module must degrade to
no-skills, not crash startup). Re-exports the discovery model, the semantic
index, the embedding backends, the ``skill://`` resolver, the
``SkillResolver`` adapter factory, and the default root computation.
Root precedence: project-local beats global, native beats ecosystem.
``default_skill_roots`` walks up from cwd to the filesystem root collecting
``.local-operator/skills`` directories (regardless of ``$HOME``), appends
``~/.local-operator/skills``, then the wider ecosystem roots
(``~/.omp/agent/skills``, ``~/.claude/skills``, ``~/.codex/skills``,
``~/.agents/skills``) last — only the ones that exist, so a clean machine
scans nothing extra. ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` replaces that
ecosystem set (colon-separated absolute paths; an empty value disables it).
Earlier roots win name collisions in :func:`discover_skills`, which is what
makes native roots authoritative over imported ones.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

from local_operator.skills.discovery import Skill, discover_skills
from local_operator.skills.embeddings import (
    ApiEmbedder,
    EmbeddingBackend,
    EmbeddingError,
    LocalEmbedder,
    default_backend_from_env,
)
from local_operator.skills.index import SkillIndex, render_block
from local_operator.skills.protocol import resolve_skill_url

__all__ = [
    "ApiEmbedder",
    "EmbeddingBackend",
    "EmbeddingError",
    "LocalEmbedder",
    "Skill",
    "SkillIndex",
    "default_backend_from_env",
    "default_skill_roots",
    "discover_skills",
    "make_skill_resolver",
    "render_block",
    "resolve_skill_url",
]

_SKILLS_SUBDIR = Path(".local-operator") / "skills"

#: Ecosystem skill directories scanned AFTER the native roots, widest first
#: is irrelevant — only existence and order matter, and order is fixed here
#: so two machines with the same installs compute the same root list.
_ECOSYSTEM_SKILL_SUBDIRS: tuple[Path, ...] = (
    Path(".omp") / "agent" / "skills",
    Path(".claude") / "skills",
    Path(".codex") / "skills",
    Path(".agents") / "skills",
)

#: Replaces the ecosystem set when set. Colon-separated absolute paths;
#: an empty value disables ecosystem scanning entirely (a user who wants
#: ONLY native roots gets exactly that).
_EXTRA_ROOTS_ENV = "LOCAL_OPERATOR_SKILL_EXTRA_ROOTS"


def _ecosystem_roots(home: Path) -> list[Path]:
    """Existing ecosystem roots, or the env override, native-root-free.

    Missing default roots are filtered here (not left for
    :func:`discover_skills` to skip) because this list is also the user-facing
    answer to "what am I scanning"; env-provided paths are filtered the same
    way so the override behaves like the set it replaces.
    """
    raw = os.environ.get(_EXTRA_ROOTS_ENV)
    if raw is None:
        candidates = [home / subdir for subdir in _ECOSYSTEM_SKILL_SUBDIRS]
    else:
        candidates = [Path(part).expanduser() for part in raw.split(":") if part.strip()]
    return [candidate for candidate in candidates if candidate.is_dir()]


def default_skill_roots(cwd: Path | None = None) -> list[Path]:
    """Compute the default discovery roots for a working directory.

    Walks up from ``cwd`` to the filesystem root collecting
    ``<dir>/.local-operator/skills`` — regardless of ``$HOME``, so a repo at
    ``/opt``, ``/srv`` or ``/Volumes`` still picks up its project-local
    roots — then appends ``~/.local-operator/skills``, then the ecosystem
    roots (see :data:`_ECOSYSTEM_SKILL_SUBDIRS` / :data:`_EXTRA_ROOTS_ENV`).
    Roots are deduped by realpath, first occurrence wins — the walk-up order
    gives project-local roots priority over global ones, and native roots
    priority over ecosystem ones, matching the collision rule in
    :func:`discover_skills`.
    """
    start = (cwd or Path.cwd()).resolve()
    home = Path.home().resolve()

    candidates: list[Path] = []
    # Walk cwd → filesystem root; every ancestor gets a candidate, so
    # project-local roots work even outside $HOME.
    current = start
    while True:
        candidates.append(current / _SKILLS_SUBDIR)
        if current.parent == current:  # reached "/"
            break
        current = current.parent
    # Global native root, then ecosystem roots: the deepest project root
    # wins collisions, and native beats imported.
    candidates.append(home / _SKILLS_SUBDIR)
    candidates.extend(_ecosystem_roots(home))

    seen: set[str] = set()
    roots: list[Path] = []
    for candidate in candidates:
        try:
            key = str(candidate) if not candidate.exists() else str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        roots.append(candidate)
    return roots


def make_skill_resolver(skills: Mapping[str, Skill]) -> Callable[[str], str | None]:
    """Build the ``SkillResolver`` adapter for the ``read`` tool.

    Contract (docs/REWRITE.md §C, ``harness/types.py``
    ``resolve_internal_url``): a resolver returns content for skill URLs,
    ``None`` for non-skill URLs (caller chains other resolvers), and never
    raises. :func:`resolve_skill_url` itself raises ``ValueError`` for
    unknown names and unsafe paths; this adapter catches it and returns the
    message AS CONTENT, so the available-names list reaches the model as a
    clean tool result and it can self-correct with a retry.
    """

    def resolver(url: str) -> str | None:
        if not url.startswith("skill://"):
            return None
        try:
            return resolve_skill_url(url, skills)
        except (ValueError, OSError) as exc:
            # OSError: a SKILL.md deleted or chmod-000'd between discovery and
            # the read. The adapter's contract is "never raises".
            return str(exc)

    return resolver

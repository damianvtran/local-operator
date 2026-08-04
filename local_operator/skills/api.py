"""Public surface of the skills subsystem (stream C).

Session and tools import ONLY from here — the lazy-import contract the
session relies on (a missing/broken skills module must degrade to
no-skills, not crash startup). Re-exports the discovery model, the semantic
index, the embedding backends, the ``skill://`` resolver, the
``SkillResolver`` adapter factory, and the default root computation.

Root precedence: project-local beats global. ``default_skill_roots`` walks
up from cwd to the filesystem root collecting ``.local-operator/skills``
directories (regardless of ``$HOME``) and appends
``~/.local-operator/skills`` last; earlier roots win name collisions in
:func:`discover_skills`.
"""

from __future__ import annotations

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


def default_skill_roots(cwd: Path | None = None) -> list[Path]:
    """Compute the default discovery roots for a working directory.

    Walks up from ``cwd`` to the filesystem root collecting
    ``<dir>/.local-operator/skills`` — regardless of ``$HOME``, so a repo at
    ``/opt``, ``/srv`` or ``/Volumes`` still picks up its project-local
    roots — then appends ``~/.local-operator/skills`` last. Roots are
    deduped by realpath, first occurrence wins — the walk-up order gives
    project-local roots priority over global ones, matching the collision
    rule in :func:`discover_skills`.
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
    # Global root is last: the deepest project root wins collisions.
    candidates.append(home / _SKILLS_SUBDIR)

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
        except ValueError as exc:
            return str(exc)

    return resolver

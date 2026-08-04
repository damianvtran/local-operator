"""Skills subsystem: discovery, semantic selection, ``skill://`` protocol."""

from local_operator.skills.api import (
    ApiEmbedder,
    EmbeddingBackend,
    EmbeddingError,
    LocalEmbedder,
    Skill,
    SkillIndex,
    default_backend_from_env,
    default_skill_roots,
    discover_skills,
    make_skill_resolver,
    render_block,
    resolve_skill_url,
)

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

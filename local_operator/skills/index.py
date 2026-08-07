"""Semantic skill index: embed skill descriptions, select the relevant ones.

This is the deliberate divergence from the common static approach, which lists ALL skill
descriptions in the system prompt every turn. Here, each turn embeds the user
message and injects ONLY the top-k descriptions scoring above a threshold
(docs/REWRITE.md §C). The prompt cost becomes O(relevant skills) instead of
O(total skills), and the volatile skills block stays small.

Layout:

- ``build()`` embeds ``"<name>: <description>"`` per skill into one
  :class:`~local_operator.skills.vectors.VectorMatrix` built ONCE and reused
  for every ``select()`` (vectors are L2-normalized, so inner product ==
  cosine).
- Vectors persist at ``cache_dir/<identity>.skills.vec`` with a matching
  ``<identity>.meta.json``; the identity digest covers the skill roots and
  the backend (class, model, base_url — not dim; see ``_identity_digest``)
  so different projects and different embedding models never share a file.
  Meta records the content
  hash over the ORDERED ``(name, description, file_path, mtime)`` sequence
  plus the exact name order; both must match on load (miss → rebuild). The
  content hash is ALSO embedded inside the vector blob and re-verified after
  load — meta and vectors can only ever describe each other.
- ``select()`` never breaks startup: if the primary backend raises, it
  degrades to the offline :class:`LocalEmbedder` first (memoized for the
  index lifetime); only if the local fallback also fails does it return ALL
  non-hidden skills (the static behavior), recording a warning.

Concurrency model: ONE process owns a cache directory at a time. Writes are
atomic per file (temp file + ``os.replace``, meta last) and the embedded hash
catches any interleaved leftover, but two processes racing on one directory
can still ping-pong rebuilds — sessions get private cache dirs for this
reason.

``hide`` skills are excluded from listings but NOT from reads — the
``skill://`` protocol resolves them regardless of selection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from local_operator.skills.discovery import Skill
from local_operator.skills.embeddings import EmbeddingBackend, LocalEmbedder
from local_operator.skills.vectors import VectorMatrix, deserialize

logger = logging.getLogger(__name__)


def _content_hash(skills: Sequence[Skill]) -> str:
    """Stable identity of the indexed content, ORDER-sensitive.

    Covers ``(name, description, file_path, mtime)`` in LIST ORDER: matrix
    rows are positional, so the same skill set in a different order is a
    different index. Editing a SKILL.md, renaming a skill, swapping roots,
    or reordering invalidates the cache; a pure rescan of identical content
    in identical order reuses it.

    """
    digest = hashlib.sha256()
    for skill in skills:
        mtime = skill.file_path.stat().st_mtime_ns if skill.file_path.exists() else 0
        entry = (
            skill.name,
            skill.description,
            str(skill.file_path),
            mtime,
        )
        digest.update(repr(entry).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _identity_digest(skills: Sequence[Skill], backend: EmbeddingBackend) -> str:
    """Cache-file identity: skill roots + backend kind.

    Keying filenames by this digest means different projects (different
    root sets) and different backends/models each get their own cache file
    instead of evicting each other on every switch. Dim is deliberately NOT
    part of the filename: a learned backend reports 0 until its first
    response, which would otherwise rename the file between the load check
    and the persist. Width mismatches are caught by meta + shape + the
    embedded hash instead.
    """
    digest = hashlib.sha256()
    roots = sorted({str(Path(skill.source).resolve()) for skill in skills})
    for root in roots:
        digest.update(root.encode("utf-8"))
        digest.update(b"\n")
    digest.update(type(backend).__name__.encode("utf-8"))
    model = getattr(backend, "model", None)
    if model:
        digest.update(str(model).encode("utf-8"))
    base_url = getattr(backend, "base_url", None)
    if base_url:
        digest.update(str(base_url).encode("utf-8"))
    return digest.hexdigest()[:16]


def _backend_kind(backend: EmbeddingBackend) -> str:
    """Short class name identifying the backend's persisted identity."""
    return type(backend).__name__


def _backend_meta(backend: EmbeddingBackend) -> Mapping[str, object]:
    """The backend's persisted identity. Any mismatch on load = cache miss.

    ``model``/``base_url`` are present on ApiEmbedder; two providers with
    equal dims must not share vectors. (The matrix ``dim`` is recorded
    separately by the caller from the actual matrix width, which a learned
    backend may only know after its first response.)
    """
    meta: dict[str, object] = {
        "backend": _backend_kind(backend),
        "model": getattr(backend, "model", None),
        "base_url": getattr(backend, "base_url", None),
    }
    return meta


def render_block(skills: list[Skill]) -> str:
    """Render the volatile ``<skills>`` system-prompt block.

    Exact wire format — one ``- name: description`` line per skill inside
    a ``<skills>`` tag, preceded by the hard "MUST read before proceeding"
    rule. Returns ``""`` for an empty list so the volatile block disappears
    entirely rather than emitting an empty tag.
    """
    if not skills:
        return ""
    lines = [
        "Skills are specialized knowledge. If one matches your task, you MUST read "
        "`skill://<name>` before proceeding.",
        "<skills>",
    ]
    lines.extend(f"- {skill.name}: {skill.description}" for skill in skills)
    lines.append("</skills>")
    return "\n".join(lines)


class SkillIndex:
    """Vector index over discovered skills with a persistent vector cache.

    The index is positional: matrix row ``i`` corresponds to
    ``self.skills[i]``. ``build()`` is idempotent — it loads the persisted
    vectors when the content hash AND name order AND backend identity match
    and rebuilds otherwise (including when the backend kind/dim/model
    recorded in the cache differs, e.g. switching from LocalEmbedder to
    ApiEmbedder or between two equal-dim API models).
    """

    def __init__(
        self,
        skills: list[Skill],
        backend: EmbeddingBackend,
        cache_dir: Path | None = None,
    ) -> None:
        self.skills = list(skills)
        self.backend = backend
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path("~/.local-operator/cache").expanduser()
        )
        self._matrix: VectorMatrix | None = None
        self._matrix_backend: EmbeddingBackend | None = None  # who embedded it
        self._warnings: list[str] = []
        self._degraded = False
        self._backend_failed = False  # memoize primary-backend failure
        self.backend_failures = 0  # count of primary-backend failures seen
        self._local_fallback: LocalEmbedder | None = None  # one stable instance

    @property
    def warnings(self) -> list[str]:
        """Diagnostics accumulated by build/select (surfaced to the user)."""
        return list(self._warnings)

    @property
    def degraded(self) -> bool:
        """True when the last select fell back to static listing."""
        return self._degraded

    # Thin alias so session code holding only a SkillIndex can render
    # without importing internals — the shared contract names it on the
    # index surface. Module-level render_block is defined above.
    render_block = staticmethod(render_block)

    # --- build / cache -----------------------------------------------------

    def _cache_stem(self) -> str:
        return f"{_identity_digest(self.skills, self.backend)}.skills"

    def _cache_vectors_path(self) -> Path:
        return self.cache_dir / f"{self._cache_stem()}.vec"

    def _cache_meta_path(self) -> Path:
        return self.cache_dir / f"{self._cache_stem()}.meta.json"

    async def build(self, backend: EmbeddingBackend | None = None) -> None:
        """Embed all skills, using the persisted cache when it matches.

        ``backend`` defaults to the index's own backend; the degradation
        ladder passes the local fallback here when the primary fails. Cache
        load/persist only happens for the index's own backend — a fallback
        build serves in-memory and is never written under a foreign identity.
        """
        backend = backend or self.backend
        own_backend = backend is self.backend
        if not self.skills:
            self._matrix = VectorMatrix.zeros(0, max(backend.dim, 1))
            return

        content_hash = _content_hash(self.skills)
        if own_backend and self._try_load_cache(content_hash):
            return

        texts = [f"{skill.name}: {skill.description}" for skill in self.skills]
        vectors = await backend.embed(texts)
        if len(vectors) != len(self.skills):
            raise RuntimeError(
                f"Embedding backend returned {len(vectors)} vectors for "
                f"{len(self.skills)} skills"
            )
        matrix = VectorMatrix.from_vectors(vectors)
        synthetic = False
        if matrix.dim == 0:
            # Backend returned junk (e.g. empty strings); keep a zero matrix
            # so search degrades to "nothing matches" rather than crashing.
            matrix = VectorMatrix.zeros(len(self.skills), backend.dim)
            synthetic = True
        self._matrix = matrix
        self._matrix_backend = backend
        # A synthetic placeholder must NOT be cached under the real content
        # hash: persisted, it reloads forever and silently degrades semantic
        # selection until the file is deleted by hand.
        if own_backend and not synthetic:
            self._persist_cache(content_hash, matrix)

    def _try_load_cache(self, content_hash: str) -> bool:
        meta_path = self._cache_meta_path()
        vectors_path = self._cache_vectors_path()
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(meta, dict):
            return False
        # Backend identity: class + the recorded dim (the actual matrix
        # width, which learned backends only know after embedding) + model
        # + base_url. Any mismatch is a miss.
        if (
            meta.get("hash") != content_hash
            or meta.get("order") != [skill.name for skill in self.skills]
            or meta.get("backend") != _backend_kind(self.backend)
            or meta.get("model") != getattr(self.backend, "model", None)
            or meta.get("base_url") != getattr(self.backend, "base_url", None)
        ):
            return False
        try:
            matrix, stored_hash = deserialize(vectors_path.read_bytes())
        except Exception:  # noqa: BLE001 — truncated/corrupt blob → rebuild
            return False
        # Hash embedded in the blob itself: meta and vectors can only ever
        # describe each other, never an interleaved neighbor's payload.
        if stored_hash != content_hash:
            return False
        if meta.get("dim") != matrix.dim or len(matrix) != len(self.skills):
            return False
        # A backend that DECLARES its width (pre-configured ApiEmbedder)
        # must not load a cache of a different width; learned backends
        # report 0 and accept whatever consistent width the cache holds.
        declared = getattr(self.backend, "dim", 0)
        if declared and matrix.dim != declared:
            return False
        self._matrix = matrix
        self._matrix_backend = self.backend
        return True

    def _persist_cache(self, content_hash: str, matrix: VectorMatrix) -> None:
        """Atomic write: vectors → temp → ``os.replace``, meta last.

        A reader either sees the old pair or the new pair, never a meta
        pointing at a half-written vector file (meta lands last, and the
        hash embedded in the blob is the final cross-check on load).
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            vectors_path = self._cache_vectors_path()
            tmp_vectors = vectors_path.with_name(vectors_path.name + ".tmp")
            tmp_vectors.write_bytes(matrix.serialize(content_hash))
            os.replace(tmp_vectors, vectors_path)

            meta = dict(_backend_meta(self.backend))
            meta.update(
                {
                    "hash": content_hash,
                    "order": [skill.name for skill in self.skills],
                    "dim": matrix.dim,
                    "count": len(self.skills),
                }
            )
            meta_path = self._cache_meta_path()
            tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
            tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
            os.replace(tmp_meta, meta_path)
        except OSError as exc:
            self._warnings.append(f"Could not persist skill index cache: {exc}")

    # --- select ------------------------------------------------------------

    async def select(self, query: str, k: int = 8, threshold: float | None = None) -> list[Skill]:
        """Return the top-k skills whose descriptions match ``query``.

        Cosine search over L2-normalized vectors (one exhaustive inner-product
        scan of the cached matrix). ``threshold`` defaults to the backend's
        ``default_threshold``. Hidden skills never appear — they stay
        reachable only via direct ``skill://`` reads. Results are sorted by
        the discovery key (name.lower, name, path) so two turns selecting
        the SAME set render a byte-identical block.

        Degradation ladder: primary backend failure → offline
        :class:`LocalEmbedder`, memoized for the index lifetime so an API
        outage costs one failed request, not one per turn (subsequent turns
        skip straight to the local fallback); local failure too → ALL
        non-hidden skills (the static behavior) with a warning.
        Selection never breaks a session.
        """
        self._degraded = False
        if not self.skills:
            return []

        if not self._backend_failed:
            picked = await self._select_with_backend(self.backend, query, k, threshold)
            if picked is not None:
                return picked

        # Primary backend failed. Fall back to the offline embedder first —
        # a worse backend beats no selection, and full listing is exactly
        # what blows the start-of-session token budget.
        if not isinstance(self.backend, LocalEmbedder):
            if not self._backend_failed:
                self._backend_failed = True
                self.backend_failures += 1
                self._warnings.append(
                    f"Embedding backend failed ({self.backend_failures}x); "
                    "falling back to the local embedder"
                )
            if self._local_fallback is None:
                self._local_fallback = LocalEmbedder()
            picked = await self._select_with_backend(self._local_fallback, query, k, threshold)
            if picked is not None:
                return picked

        self._degraded = True
        if not getattr(self, "_degraded_warned", False):
            self._degraded_warned = True
            self._warnings.append("Skill selection degraded to static listing")
        return [skill for skill in self.skills if not skill.hide]

    async def _select_with_backend(
        self,
        backend: EmbeddingBackend,
        query: str,
        k: int,
        threshold: float | None,
    ) -> list[Skill] | None:
        """Score with ``backend``; None on any failure (caller degrades)."""
        if threshold is None:
            threshold = backend.default_threshold
        try:
            if self._matrix is None or self._matrix_backend is not backend:
                await self.build(backend)
            query_vec = (await backend.embed([query]))[0]
            scores = self._scores(query_vec)
        except Exception as exc:  # noqa: BLE001 — degradation is the contract
            logger.warning("Skill selection failed: %s", exc)
            return None

        ranked = sorted(range(len(self.skills)), key=lambda i: scores[i], reverse=True)
        picked: list[Skill] = []
        for i in ranked:
            if scores[i] < threshold:
                break  # ranked descending: everything after is below too
            skill = self.skills[i]
            if skill.hide:
                continue
            picked.append(skill)
            if len(picked) >= k:
                break
        picked.sort(key=lambda s: (s.name.lower(), s.name, str(s.file_path)))
        return picked

    def _scores(self, query_vec: list[float]) -> list[float]:
        """Cosine similarities of the query against every indexed skill.

        One exhaustive inner-product scan of the cached matrix, returned in
        skill order so the caller can filter by threshold and ``hide`` in a
        single pass. Exhaustive is the right shape here: the threshold is
        absolute, so every row has to be scored anyway (see
        :mod:`local_operator.skills.vectors`).
        """
        matrix = self._matrix
        assert matrix is not None
        # build() and _try_load_cache() both refuse a matrix whose row count
        # disagrees with the skill list, so positions line up by construction.
        assert len(matrix) == len(self.skills)
        return matrix.scores(query_vec)

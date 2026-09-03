"""Semantic index for skills, packaged guides, and private routing hints.

Only short names and descriptions selected above an absolute similarity
threshold enter the system tail. Full skill and guide bodies stay behind their
respective URL protocols. Registered-agent hints share the vector machinery but
are never rendered; they can surface the generic agents guide without exposing
the registry itself.

Scoring is hybrid: cosine over embeddings, blended with a lexical Jaccard
over ``name: description`` (see :func:`_hybrid_scores`), and gitignore-style
``globs`` frontmatter force-includes path-matched skills at 1.0. The
threshold applies to that final score.

Vectors persist at ``cache_dir/<identity>.skills.vec``. The content hash covers
the ordered resource type, name, description, file path, and mtime; the matrix
row order and backend identity must also match before a cache is reused.
Selection degrades from the configured embedding backend to the deterministic
local embedder, then to a static listing. Hidden resources remain directly
readable but never appear in semantic results.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from local_operator.skills.discovery import Skill
from local_operator.skills.embeddings import EmbeddingBackend, LocalEmbedder
from local_operator.skills.protocol import resource_url
from local_operator.skills.vectors import VectorMatrix, deserialize

logger = logging.getLogger(__name__)

#: Word tokenizer for the lexical half of the hybrid score. Deliberately the
#: same shape as the local embedder's (``[a-z0-9]+`` on lowercased text) so
#: both halves of the blend see the same words.
_WORD_RE = re.compile(r"[a-z0-9]+")

#: Hybrid blend weights: ``final = max(cos, w_c*cos + w_l*jaccard)``. The
#: outer max keeps a strong semantic match from being dragged down by a
#: near-zero Jaccard (a short query against a long description), while the
#: blend alone lifts a lexically-exact hit whose embedding landed below
#: threshold.
_COSINE_WEIGHT = 0.6
_LEXICAL_WEIGHT = 0.4

#: Punctuation stripped from query tokens before path-likeness is judged —
#: "fix (src/app.py)" must yield "src/app.py", not "(src/app.py)".
_TOKEN_STRIP = "\"'`()[]{}<>,;:"

#: A bare filename with an extension ("pyproject.toml"). Must end in
#: alphanumerics, so sentence noise like "e.g." or "3.2." is rejected while
#: real filenames pass.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_+.~\-]+\.[A-Za-z0-9]+$")


#: Markers of an authentication failure in an embedding backend's exception.
#: Matched against the exception's class name + text because the embedding
#: backends raise their own types (httpx errors, provider-specific wrappers),
#: and importing the provider stack to classify one degraded lookup would put
#: that whole graph on the skill-index path. Deliberately narrow: only the
#: unambiguous 401/invalid-key wordings, so a transient 5xx or a timeout still
#: surfaces its warning to the user.
_AUTH_FAILURE_MARKERS = (
    "401",
    "unauthorized",
    "invalid api key",
    "invalid api-key",
    "invalid x-api-key",
    "authentication",
    "no api key",
    "missing api key",
)


def _looks_like_auth_failure(exc: BaseException) -> bool:
    """Best-effort: did ``exc`` come from a rejected/absent credential?

    Used only to decide whether the "embedding backend failed" warning belongs
    in the log rather than in front of the user (the credential problem is
    already reported elsewhere). Being wrong is cheap in both directions: a
    false positive hides one redundant warning, a false negative shows it.
    """
    haystack = f"{type(exc).__name__}: {exc}".lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (401, 403):
        return True
    return any(marker in haystack for marker in _AUTH_FAILURE_MARKERS)


def _path_suffixes(path: str) -> Sequence[str]:
    """A path plus every component-suffix of it: ``a/b/c`` → ``a/b/c``,
    ``b/c``, ``c``.

    Suffix matching is what makes an unanchored gitignore pattern behave at
    any depth (``*.py`` must hit ``…/src/app.py``) without knowing where the
    repository root is — selection only ever sees absolute paths.
    """
    suffixes = [path]
    parts = [part for part in path.split("/") if part]
    suffixes.extend("/".join(parts[start:]) for start in range(1, len(parts)))
    return suffixes


def _query_path_tokens(query: str) -> list[str]:
    """Extract file-path-like tokens from a natural-language query.

    "fix the import in src/app/main.py please" yields ``src/app/main.py``.
    Path-like means: contains a slash, starts with ``.``/``~`` (relative or
    home paths, dotfiles), or is a bare filename with an extension. Prose
    words are ignored so ordinary queries cannot trigger path routing.
    """
    tokens: list[str] = []
    for raw in query.split():
        token = raw.strip(_TOKEN_STRIP)
        if token in ("", ".", ".."):
            continue
        if "/" in token or token.startswith((".", "~")) or _FILENAME_RE.match(token):
            tokens.append(token)
    return tokens


def _glob_matches(globs: Sequence[str], cwd: Path | None, query: str) -> bool:
    """True when any gitignore-style glob matches the cwd or a query token.

    Matching is fnmatch against each target and every component-suffix of it
    (see :func:`_path_suffixes`), which reproduces gitignore's any-depth
    behavior for unanchored patterns. A leading ``/`` anchor is dropped —
    redundant once suffixes are tried — and a trailing ``/`` (directory-only)
    is dropped too, because targets are never stat'ed here; an author who
    needs that precision should write the distinguishing path components.
    """
    patterns = [pattern.strip().strip("/") for pattern in globs]
    patterns = [pattern for pattern in patterns if pattern]
    if not patterns:
        return False
    targets: list[str] = []
    if cwd is not None:
        targets.append(str(cwd))
    targets.extend(_query_path_tokens(query))
    return any(
        fnmatch.fnmatchcase(candidate, pattern)
        for pattern in patterns
        for target in targets
        for candidate in _path_suffixes(target)
    )


def _hybrid_scores(
    query: str,
    skills: Sequence[Skill],
    cosines: Sequence[float],
    cwd: Path | None,
) -> list[float]:
    """Blend cosine similarity with lexical Jaccard, then apply glob boosts.

    The shipped local embedder is honestly ~71% recall — it hashes character
    n-grams, so it IS lexical — and the exact-word overlap the embedder
    washes out is recoverable for free with a Jaccard over the same
    ``name: description`` strings the index already embeds. The lexical
    blend is the free part of that fix: no new backend, no re-embedding,
    deterministic, byte-stable across turns.

    ``final = max(cos, 0.6*cos + 0.4*jaccard)``; glob-matched skills are
    forced to 1.0, bypassing the cosine threshold entirely (the author's
    path claim outranks the embedder's guess) while remaining subject to
    the caller's ``k`` cap and the hidden filter downstream.
    """
    query_words = frozenset(_WORD_RE.findall(query.lower()))
    scores: list[float] = []
    for skill, cosine in zip(skills, cosines):
        if _glob_matches(skill.globs, cwd, query):
            scores.append(1.0)
            continue
        text_words = frozenset(_WORD_RE.findall(f"{skill.name}: {skill.description}".lower()))
        union = len(query_words | text_words)
        jaccard = len(query_words & text_words) / union if union else 0.0
        scores.append(max(cosine, _COSINE_WEIGHT * cosine + _LEXICAL_WEIGHT * jaccard))
    return scores


def _content_hash(skills: Sequence[Skill]) -> str:
    """Stable identity of indexed resources, including their protocol kind.

    Rows are positional. Editing a body, changing a routing description,
    moving a resource between ``skill``/``guide``/``agent_hint``, or reordering
    the list must invalidate the matrix.
    """
    digest = hashlib.sha256()
    for skill in skills:
        mtime = skill.file_path.stat().st_mtime_ns if skill.file_path.exists() else 0
        entry = (
            skill.resource_type,
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
    """Render selected guides and skills without rendering private hints.

    The stable base prompt still owns the ``guide://`` protocol rule in full;
    the one-line imperative here is the local reminder that a selected guide is
    meant to be READ. Selected skills use concrete protocol URLs rather than a
    placeholder because the router has already established which ones match.
    """
    guides = [item for item in skills if item.resource_type == "guide"]
    user_skills = [item for item in skills if item.resource_type == "skill"]
    sections: list[str] = []
    if guides:
        # The imperative mirrors the skills branch below. Without it this
        # section was bare name/description lines and the only instruction to
        # act on them lived far away in the base prompt, phrased as a
        # conditional about questions on Local Operator itself - which a model
        # asked to DO something (rather than answer a question) does not
        # classify as a match. Costs its tokens only when a guide is selected.
        lines = [
            "Guides are procedures for this harness. If one matches your task, you MUST "
            "read `guide://<name>` before acting, even if you think you already know the "
            "answer.",
            "<guides>",
        ]
        lines.extend(f"- {guide.name}: {guide.description}" for guide in guides)
        lines.append("</guides>")
        sections.append("\n".join(lines))
    if user_skills:
        # Naming the already-selected resources removes the inference step that
        # led models to search the filesystem for SKILL.md instead of using the
        # virtual resource protocol. This cost stays local to selected skills.
        selected_urls = ", ".join(f"`{resource_url('skill', skill.name)}`" for skill in user_skills)
        lines = [
            (
                "Skills provide domain-specific instructions and workflows. Read these selected "
                f"skills immediately before proceeding: {selected_urls}. Do not search the "
                "filesystem or use bash/glob to locate skills — skills are virtual resources "
                "loaded only via `skill://`. The skill body ends with its reference files; read "
                "those with `skill://<name>/<path>`, never a raw filesystem path."
            ),
            "<skills>",
        ]
        lines.extend(f"- {skill.name}: {skill.description}" for skill in user_skills)
        lines.append("</skills>")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


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
        #: Whether the most recent backend failure looked like an auth error
        #: (401/invalid key). Used to route the "backend failed" warning to the
        #: log instead of the user when the cause is the credential problem the
        #: app already reports — see the fallback branch in :meth:`select`.
        self._last_backend_auth_failure = False

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
        """Embed all routable resources, using the persisted cache when it matches.

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
                f"{len(self.skills)} resources"
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

    async def select(
        self,
        query: str,
        k: int = 8,
        threshold: float | None = None,
        *,
        cwd: Path | None = None,
    ) -> list[Skill]:
        """Return the top-k skills whose descriptions match ``query``.

        Cosine search over L2-normalized vectors (one exhaustive inner-product
        scan of the cached matrix), blended with a lexical Jaccard and
        gitignore-style glob boosts (see :func:`_hybrid_scores`); the
        threshold applies to that final score. ``cwd`` is the session working
        directory, matched alongside path-like query tokens for globs.
        Hidden skills never appear — they stay reachable only via direct
        ``skill://`` reads. Results are sorted by the discovery key
        (name.lower, name, path) so two turns selecting the SAME set render
        a byte-identical block.

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
            picked = await self._select_with_backend(self.backend, query, k, threshold, cwd)
            if picked is not None:
                return picked

        # Primary backend failed. Fall back to the offline embedder first —
        # a worse backend beats no selection, and full listing is exactly
        # what blows the start-of-session token budget.
        if not isinstance(self.backend, LocalEmbedder):
            if not self._backend_failed:
                self._backend_failed = True
                self.backend_failures += 1
                note = (
                    f"Embedding backend failed ({self.backend_failures}x); "
                    "falling back to the local embedder"
                )
                # When the cause is the SAME missing/invalid credential the app
                # already warns about elsewhere (a 401 from the embedding
                # provider), surfacing a second "embedding backend failed"
                # warning is noise proportional to nothing — the user has one
                # credential problem, not two — and it accuses a subsystem that
                # is working exactly as designed (it degraded to the local
                # embedder). Downgrade that specific case to the log file; every
                # other backend failure still surfaces.
                if self._last_backend_auth_failure:
                    logger.info("%s (cause: provider authentication)", note)
                else:
                    self._warnings.append(note)
            if self._local_fallback is None:
                self._local_fallback = LocalEmbedder()
            picked = await self._select_with_backend(self._local_fallback, query, k, threshold, cwd)
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
        cwd: Path | None = None,
    ) -> list[Skill] | None:
        """Score with ``backend``; None on any failure (caller degrades)."""
        if threshold is None:
            threshold = backend.default_threshold
        try:
            if self._matrix is None or self._matrix_backend is not backend:
                await self.build(backend)
            query_vec = (await backend.embed([query]))[0]
            scores = self._scores(query_vec)
            scores = _hybrid_scores(query, self.skills, scores, cwd)
        except Exception as exc:  # noqa: BLE001 — degradation is the contract
            # Remember whether THIS failure was an auth error, so the caller can
            # decide the "backend failed" warning belongs in the log rather than
            # in front of the user (see the fallback branch in select). Read
            # best-effort — the embedding backends raise their own exception
            # types, so classify from status/text markers without importing the
            # provider stack onto this path.
            self._last_backend_auth_failure = _looks_like_auth_failure(exc)
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

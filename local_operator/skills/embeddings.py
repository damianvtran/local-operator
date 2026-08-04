"""Embedding backends for semantic skill selection.

Two implementations behind one Protocol:

- :class:`LocalEmbedder` — deterministic, offline, dependency-free. Hashed
  character 3-gram + 4-gram TF vectors (blake2b → bucket, sublinear TF
  weights, stopword removal, light suffix stemming, L2 normalized). It is
  deliberately crude: skill descriptions are short, vocabulary overlap is
  the dominant signal, and correctness here means "same text → same vector,
  every run, on every machine" — no model download, no network, no
  nondeterminism. Always available as the fallback backend.
- :class:`ApiEmbedder` — OpenAI-compatible ``POST {base}/embeddings`` via an
  injected ``httpx.AsyncClient`` (tests use ``MockTransport``; the session
  passes a real client). Used when an embeddings-capable API key is
  configured.

``default_threshold`` lives on the backend because the two backends live on
different similarity scales. The local threshold is NOT a guess: it is the
midpoint of the measured gap between matching and unrelated scores on the
calibration corpus in ``tests/unit/skills/test_calibration.py`` (median
unrelated-pair cosine ≈ 0.07, max unrelated query score ≈ 0.15, matching
query top score ≥ 0.42), pinned there by test. API embeddings separate
meanings more cleanly, so the API bar sits near the local match floor.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Protocol, runtime_checkable

import httpx


class EmbeddingError(RuntimeError):
    """Raised by backends when embedding fails; index degrades gracefully."""


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that can turn texts into L2-normalized vectors.

    Contract: returns one vector per input text, in order. Vectors SHOULD be
    L2-normalized so inner product equals cosine similarity (both bundled
    backends guarantee this). ``default_threshold`` is the minimum cosine
    score for a skill description to be injected into the prompt. ``dim`` is
    the vector width; backends that learn it lazily (ApiEmbedder) report 0
    until the first response arrives.
    """

    dim: int
    default_threshold: float

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in order; raises :class:`EmbeddingError` on failure."""
        ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


#: High-frequency English function words. Pure TF weights them as heavily as
#: content words, which inflates the cosine between ANY two English texts
#: (measured: median unrelated-pair cosine 0.31 at dim 512 before removal).
#: Dropping them from the gram pool is a fixed, corpus-independent stand-in
#: for IDF — a single query cannot compute document frequencies.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on for with without at by from as
    is are was were be been being it its this that these those i you he she we
    they them his her their our your my me us not no so do does did done can
    could will would should shall may might must have has had having when where
    who whom what which how why all any both each few more most other some such
    only own same into over under again further once here there about between
    out up down off above below through during before after use using uses used
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Ordered longest-first so "settings" strips "ings", not "s". Crude on
#: purpose: it only needs "testing/tests/tested" to share grams with "test".
_SUFFIXES = ("ings", "ing", "ies", "ed", "es", "ly", "s")


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


class LocalEmbedder:
    """Hashed char 3+4-gram TF embedder: offline, deterministic, dim=4096.

    Per text: lowercase, tokenize on word characters, drop stopwords, strip
    one crude inflection suffix per token, then for BOTH n=3 and n=4 extract
    every character n-gram, weight it with sublinear TF ``1 + log(tf)``, and
    hash it with ``blake2b(digest_size=4)`` into a bucket ``% dim``. Each n
    is L2-normalized separately, the two are averaged 50/50 and the result
    is re-normalized — the 4-gram channel halves the collision-driven and
    common-English noise floor that pure 3-gram TF carries, without erasing
    the 3-gram recall on short tokens.

    Calibration (``tests/unit/skills/test_calibration.py`` is the contract):
    over an 8-skill corpus, median unrelated-pair cosine ≈ 0.07, a clearly
    matching query scores ≥ 0.42 on its skill while its best unrelated score
    stays ≤ 0.15. ``default_threshold`` is the shipped midpoint of that gap.

    Empty/short texts map to the zero vector, which scores 0 against
    everything — a skill with an empty routing signal is never injected.
    """

    dim: int
    default_threshold: float

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        # Shipped constant from the calibration corpus; pinned by test.
        self.default_threshold = 0.27

    def _tokens(self, text: str) -> list[str]:
        return [
            _stem(token)
            for token in _WORD_RE.findall(text.lower())
            if token not in _STOPWORDS
        ]

    def _gram_vector(self, tokens: list[str], n: int) -> list[float]:
        counts: dict[str, int] = {}
        for token in tokens:
            for i in range(len(token) - n + 1):
                gram = token[i : i + n]
                counts[gram] = counts.get(gram, 0) + 1
        vec = [0.0] * self.dim
        for gram, tf in counts.items():
            bucket = (
                int.from_bytes(
                    hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest(),
                    "little",
                )
                % self.dim
            )
            vec[bucket] += 1.0 + math.log(tf)
        return vec

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text (sync; used by tests and the async wrapper)."""
        tokens = self._tokens(text)
        mixed: list[float] | None = None
        channels = 0
        for n in (3, 4):
            part = _normalize(self._gram_vector(tokens, n))
            if not any(part):
                continue
            if mixed is None:
                mixed = part
            else:
                mixed = [a + b for a, b in zip(mixed, part)]
            channels += 1
        if mixed is None or channels == 0:
            return [0.0] * self.dim
        return _normalize(mixed)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Never fails — the whole point of the local backend."""
        return [self.embed_one(text) for text in texts]


class ApiEmbedder:
    """OpenAI-compatible ``/embeddings`` client over an injected AsyncClient.

    ``base_url`` is the provider root (e.g. ``https://api.openai.com/v1``);
    the endpoint is always ``{base_url}/embeddings``. Raises
    :class:`EmbeddingError` on any transport or API failure — the index
    catches it and degrades to the local backend, so raising is the right
    move.

    ``dim`` is optional: when omitted, the width is LEARNED from the first
    response (all rows validated uniform) — providers disagree on widths and
    a wrong declared dim used to poison the cache shape check forever. When
    given, a response of any other width raises instead of silently passing.
    """

    dim: int
    default_threshold: float

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        client: httpx.AsyncClient | None = None,
        dim: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim if dim is not None else 0
        self.default_threshold = 0.25
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
            vectors = [item["embedding"] for item in payload["data"]]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        widths = {len(vector) for vector in vectors}
        if len(widths) != 1:
            raise EmbeddingError(
                f"Embedding API returned non-uniform vector widths: {sorted(widths)}"
            )
        width = widths.pop()
        if self.dim == 0:
            self.dim = width
        elif self.dim != width:
            raise EmbeddingError(
                f"Embedding API returned width {width}, declared dim is {self.dim}"
            )
        return vectors

    async def aclose(self) -> None:
        """Close the client only when we created it (injected clients are
        owned by the caller)."""
        if self._owns_client:
            await self._client.aclose()


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def default_backend_from_env(
    get_credential: Callable[[str], str | None],
    base_url: str | None = None,
) -> EmbeddingBackend:
    """Pick the best available backend from configured credentials.

    ``get_credential`` is the credential cascade hook (env + stored keys) —
    we never read ``os.environ`` directly so tests and the auth store stay in
    control. Preference order: OPENAI_API_KEY → ApiEmbedder (explicit
    ``base_url`` or the OpenAI default); else OPENROUTER_API_KEY →
    ApiEmbedder against OpenRouter; else :class:`LocalEmbedder`.
    """
    openai_key = get_credential("OPENAI_API_KEY")
    if openai_key:
        return ApiEmbedder(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=openai_key,
        )
    openrouter_key = get_credential("OPENROUTER_API_KEY")
    if openrouter_key:
        return ApiEmbedder(
            base_url=base_url or _OPENROUTER_BASE_URL,
            api_key=openrouter_key,
        )
    return LocalEmbedder()

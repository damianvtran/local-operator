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

import asyncio
import hashlib
import math
import re
from typing import Any, Callable, Protocol, runtime_checkable

import httpx

from local_operator.providers.clients import (
    openrouter_attribution_headers,
    raise_for_status,
)
from local_operator.providers.failover import (
    ProviderError,
    backoff_delay_ms,
    is_transient_error,
    wrap_transport_error,
)

#: Same-endpoint attempts for one embedding request. Small on purpose: the
#: index degrades to :class:`LocalEmbedder` when this gives up, so the cost of
#: stopping is worse routing rather than a broken session, and this call sits on
#: the start-of-session critical path where a long retry ladder is felt as boot
#: lag. Three attempts covers the single-blip case that used to be terminal.
EMBED_MAX_ATTEMPTS = 3
EMBED_BASE_DELAY_MS = 250


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
_STOPWORDS = frozenset("""
    a an the and or but if then else of to in on for with without at by from as
    is are was were be been being it its this that these those i you he she we
    they them his her their our your my me us not no so do does did done can
    could will would should shall may might must have has had having when where
    who whom what which how why all any both each few more most other some such
    only own same into over under again further once here there about between
    out up down off above below through during before after use using uses used
    """.split())

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

    Calibration. 0.19 sits inside the INTERSECTION of the score gaps of two
    independent corpora, so it is tuned to neither alone. Both are
    reproducible — the first from the test module, the second with
    ``scripts/calibrate_skill_threshold.py --skills-dir <dir> --labelled
    scripts/calibration_queries.json``:

        corpus                              unrelated max   relevant min
        6 skills, pinned in test_calibration      0.1667         0.2606
        15 skills, external validation run        0.1499         0.2134
        ---------------------------------------------------------------
        intersection                              0.1667         0.2134

    0.19 is the midpoint, maximising the SMALLER margin (+0.0233 above the best
    false match, +0.0234 below the worst true match). On those two corpora and
    those query sets, recall is 100% and false positives 0%.

    HONEST LIMIT, because the sentence above is corpus- and query-set-specific
    and it would be easy to read as a general guarantee. Against the same real
    15-skill corpus with an INDEPENDENTLY chosen query set, recall drops to
    ~71%: a hashed term-frequency embedder matches vocabulary, not meaning, so
    a query sharing no terms with a description scores near zero however
    relevant it is. "where do i put my api keys" against a description saying
    "credential loading and storage" is the shape that fails. Three skills also
    rank behind a sibling that shares more vocabulary, which no threshold can
    fix.

    :class:`ApiEmbedder` is the accurate path and is selected automatically
    whenever an embedding key is configured. This class is the OFFLINE fallback
    and its ceiling is lexical overlap — that is the trade for needing no key,
    no network and no compiled dependency.

    A rejected idea, recorded so it is not retried blind: capping the embedded
    description at 300 chars to reduce the length dilution that L2-normalised
    TF suffers. It looks good on a small hand-picked query set (recall 57% ->
    71%) and is WORSE on the shipped labelled set (100% -> 83%, and the score
    gap disappears entirely), because trimming removes matching vocabulary as
    often as it removes noise. Measure any such change with
    ``scripts/calibration_queries.json``, not with a fresh set of queries
    chosen while looking at the failure.

    The earlier 0.27 came from an 8-skill corpus scored only with "clearly
    matching" queries that reached >= 0.42. On realistic input it returned
    NOTHING for 17% of the pinned queries and 44% of the external ones, in
    every case with the correct skill still ranked FIRST. Selecting nothing is
    the expensive failure mode: the agent proceeds without the playbook and no
    warning fires, because zero matches is a legal result.

    Empty/short texts map to the zero vector, which scores 0 against
    everything — a skill with an empty routing signal is never injected.
    """

    dim: int
    default_threshold: float

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        # Midpoint of the two-corpus intersection above; pinned by test with a
        # MINIMUM margin, not just a strict inequality. Do not move this toward
        # 0.27 on the strength of the RS-03 note below — that measurement came
        # from a dim-512 vector space whose noise floor was 0.31; at dim 4096
        # it is 0.05-0.08. Re-measure both ends if the dim changes again.
        self.default_threshold = 0.19

    def _tokens(self, text: str) -> list[str]:
        return [_stem(token) for token in _WORD_RE.findall(text.lower()) if token not in _STOPWORDS]

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
        vectors = await self._fetch(texts)
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
        # The index compares inner products against a COSINE threshold, so the
        # vectors must be unit-length. Providers differ: Ollama and many
        # OpenRouter-proxied models return norms > 1 (every skill clears the
        # bar, blowing the token budget), OpenAI's shortened embeddings
        # < 1 (nothing ever matches). Normalize here, once.
        return [_normalize(vector) for vector in vectors]

    async def _fetch(self, texts: list[str]) -> list[Any]:
        """POST once, retrying only what a retry can fix.

        Before this, ONE dropped connection or 503 ended the API backend for the
        whole session: :class:`~local_operator.skills.index.SkillIndex` memoizes
        a backend failure, so a blip on the first turn silently downgraded every
        later selection to the offline embedder. This is a provider call like
        any other and gets the harness's retry.

        The status is mapped through the wire clients' own
        :func:`~local_operator.providers.clients.raise_for_status` rather than
        ``httpx.raise_for_status`` so the classification is the SAME vocabulary
        the streaming paths use: ``httpx.HTTPStatusError`` carries its status
        only inside a prose message, which no classifier can read.

        Not retried, and each for its own reason: a rate limit needs a wait
        longer than a boot-path budget, a rejected key gives the same answer
        forever, and a malformed payload is a defect rather than weather.
        """
        for attempt in range(1, EMBED_MAX_ATTEMPTS + 1):
            last = attempt >= EMBED_MAX_ATTEMPTS
            headers: dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}
            # When proxying embedding calls through OpenRouter, provide app
            # attribution headers so requests are properly identified and ranked.
            if "openrouter.ai" in self.base_url:
                headers.update(openrouter_attribution_headers())
            try:
                response = await self._client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": texts},
                    timeout=60.0,
                )
                raise_for_status(response)
                payload = response.json()
                return [item["embedding"] for item in payload["data"]]
            except ProviderError as exc:
                if last or not is_transient_error(exc):
                    raise EmbeddingError(f"Embedding request failed: {exc}") from exc
            except httpx.TransportError as exc:
                # A connect/read/protocol failure is weather. It must be listed
                # ahead of HTTPError (its own base) and is wrapped through the
                # driver's helper rather than interpolated raw, because the
                # argumentless httpx errors stringify to nothing at all.
                if last:
                    raise EmbeddingError(
                        f"Embedding request failed: {wrap_transport_error(exc)}"
                    ) from exc
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                # A malformed response is a defect, not weather: asking again
                # gets the same unparseable answer.
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc
            await asyncio.sleep(backoff_delay_ms(EMBED_BASE_DELAY_MS, attempt) / 1000)
        raise EmbeddingError(  # unreachable: the last attempt always raises above
            f"Embedding request failed after {EMBED_MAX_ATTEMPTS} attempts"
        )

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

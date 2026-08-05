"""Embedding backend tests: LocalEmbedder determinism/norm/similarity,
ApiEmbedder wire contract, and the env-based backend picker."""

from __future__ import annotations

import math

import httpx
import pytest

from local_operator.skills.embeddings import (
    ApiEmbedder,
    EmbeddingBackend,
    EmbeddingError,
    LocalEmbedder,
    default_backend_from_env,
)


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class TestLocalEmbedder:
    def test_deterministic_across_calls_and_instances(self) -> None:
        embedder_a = LocalEmbedder()
        embedder_b = LocalEmbedder()
        text = "Python testing workflows"
        assert embedder_a.embed_one(text) == embedder_a.embed_one(text)
        assert embedder_a.embed_one(text) == embedder_b.embed_one(text)

    @pytest.mark.asyncio
    async def test_embed_batch_matches_singles(self) -> None:
        embedder = LocalEmbedder()
        texts = ["alpha", "beta", ""]
        batch = await embedder.embed(texts)
        assert batch == [embedder.embed_one(t) for t in texts]
        assert len(batch) == 3

    def test_l2_normalized(self) -> None:
        embedder = LocalEmbedder()
        vec = embedder.embed_one("pytest unit tests for python code")
        assert _norm(vec) == pytest.approx(1.0)
        assert len(vec) == 4096

    def test_empty_and_short_text_is_zero_vector(self) -> None:
        embedder = LocalEmbedder()
        assert embedder.embed_one("") == [0.0] * 4096
        assert embedder.embed_one("ab") == [0.0] * 4096  # too short for grams

    def test_similar_texts_score_higher(self) -> None:
        embedder = LocalEmbedder()
        query = embedder.embed_one("python testing")
        relevant = embedder.embed_one("pytest unit tests")
        irrelevant = embedder.embed_one("banana bread")
        assert _cos(query, relevant) > _cos(query, irrelevant)

    def test_custom_dim(self) -> None:
        embedder = LocalEmbedder(dim=64)
        vec = embedder.embed_one("some text here")
        assert len(vec) == 64
        assert _norm(vec) == pytest.approx(1.0)

    def test_threshold_defaults(self) -> None:
        # Shipped calibration constants — pinned by test_calibration.py.
        assert LocalEmbedder().default_threshold == 0.19
        assert ApiEmbedder(base_url="http://x", api_key="k").default_threshold == 0.25

    def test_satisfies_protocol(self) -> None:
        assert isinstance(LocalEmbedder(), EmbeddingBackend)


class TestApiEmbedder:
    @pytest.mark.asyncio
    async def test_success_posts_openai_compatible_request(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["payload"] = request.read()
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": [0.6, 0.8]},
                        {"embedding": [1.0, 0.0]},
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = ApiEmbedder(
            base_url="https://api.openai.com/v1", api_key="sk-test", client=client
        )
        vectors = await embedder.embed(["one", "two"])
        assert vectors == [[0.6, 0.8], [1.0, 0.0]]
        assert captured["url"] == "https://api.openai.com/v1/embeddings"
        assert captured["auth"] == "Bearer sk-test"
        assert b"one" in captured["payload"] and b"two" in captured["payload"]
        assert b"text-embedding-3-small" in captured["payload"]  # default model

    @pytest.mark.asyncio
    async def test_http_error_raises_embedding_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = ApiEmbedder(base_url="http://x/v1", api_key="k", client=client)
        with pytest.raises(EmbeddingError):
            await embedder.embed(["hello"])

    @pytest.mark.asyncio
    async def test_vector_count_mismatch_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = ApiEmbedder(base_url="http://x/v1", api_key="k", client=client)
        with pytest.raises(EmbeddingError):
            await embedder.embed(["one", "two"])

    @pytest.mark.asyncio
    async def test_empty_input_skips_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("should not be called")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = ApiEmbedder(base_url="http://x/v1", api_key="k", client=client)
        assert await embedder.embed([]) == []

    @pytest.mark.asyncio
    async def test_dim_learned_from_first_response(self) -> None:
        # RS-08: no declared dim → learn the width from the response.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"embedding": [0.5] * 768}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = ApiEmbedder(base_url="http://x/v1", api_key="k", client=client)
        assert embedder.dim == 0  # unknown until first response
        await embedder.embed(["hello"])
        assert embedder.dim == 768

    @pytest.mark.asyncio
    async def test_non_uniform_widths_raise(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"embedding": [1.0]}, {"embedding": [1.0, 0.0]}]},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = ApiEmbedder(base_url="http://x/v1", api_key="k", client=client)
        with pytest.raises(EmbeddingError, match="non-uniform"):
            await embedder.embed(["one", "two"])

    @pytest.mark.asyncio
    async def test_declared_dim_mismatch_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = ApiEmbedder(base_url="http://x/v1", api_key="k", client=client, dim=1536)
        with pytest.raises(EmbeddingError, match="declared dim"):
            await embedder.embed(["one"])


class TestDefaultBackendFromEnv:
    def test_openai_key_selects_api_embedder(self) -> None:
        creds = {"OPENAI_API_KEY": "sk-openai"}
        backend = default_backend_from_env(lambda k: creds.get(k))
        assert isinstance(backend, ApiEmbedder)
        assert backend.base_url == "https://api.openai.com/v1"

    def test_openai_key_with_custom_base_url(self) -> None:
        creds = {"OPENAI_API_KEY": "sk-openai"}
        backend = default_backend_from_env(
            lambda k: creds.get(k), base_url="http://localhost:11434/v1"
        )
        assert isinstance(backend, ApiEmbedder)
        assert backend.base_url == "http://localhost:11434/v1"

    def test_openrouter_key_selects_openrouter(self) -> None:
        creds = {"OPENROUTER_API_KEY": "or-key"}
        backend = default_backend_from_env(lambda k: creds.get(k))
        assert isinstance(backend, ApiEmbedder)
        assert "openrouter" in backend.base_url

    def test_openai_key_wins_over_openrouter(self) -> None:
        creds = {"OPENAI_API_KEY": "sk", "OPENROUTER_API_KEY": "or"}
        backend = default_backend_from_env(lambda k: creds.get(k))
        assert isinstance(backend, ApiEmbedder)
        assert "openai" in backend.base_url

    def test_no_keys_selects_local_embedder(self) -> None:
        backend = default_backend_from_env(lambda k: None)
        assert isinstance(backend, LocalEmbedder)

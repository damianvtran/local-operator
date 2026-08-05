"""SkillIndex tests: build/cache semantics, threshold + hide selection,
render_block format, and degradation when the backend fails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.skills.discovery import Skill
from local_operator.skills.embeddings import EmbeddingError, LocalEmbedder
from local_operator.skills.index import SkillIndex, render_block


def _make_skill(root: Path, dirname: str, description: str, hide: bool = False) -> Skill:
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\ndescription: {description}\n---\n# {dirname}", encoding="utf-8")
    return Skill(
        name=dirname,
        description=description,
        file_path=skill_md,
        base_dir=skill_dir,
        source=str(root),
        hide=hide,
    )


class CountingBackend:
    """Fake backend that counts embed calls — proves cache hits skip work."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.default_threshold = 0.18
        self.embed_calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        # Deterministic junk vectors; values are irrelevant to these tests.
        return [[(i + j) % 7 / 7.0 for j in range(self.dim)] for i in range(len(texts))]


class FailingBackend:
    """Always raises — models an API embedder whose provider is down."""

    dim = 8
    default_threshold = 0.18

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("backend offline")


class BrokenLocalEmbedder(LocalEmbedder):
    """A LocalEmbedder that raises — drives the last-resort static listing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("local backend broken")


def _cache_npz(cache: Path) -> Path:
    """The single identity-keyed npz a test cache dir holds."""
    npzs = list(cache.glob("*.skills.npz"))
    assert len(npzs) == 1, f"expected one keyed cache file, found {npzs}"
    return npzs[0]


class TestBuildAndCache:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_embedding(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        backend1 = CountingBackend()
        await SkillIndex(skills, backend1, cache_dir=cache).build()
        assert backend1.embed_calls == 1
        assert _cache_npz(cache).exists()
        assert list(cache.glob("*.skills.meta.json"))

        backend2 = CountingBackend()
        await SkillIndex(skills, backend2, cache_dir=cache).build()
        assert backend2.embed_calls == 0  # loaded from cache

    @pytest.mark.asyncio
    async def test_cache_miss_on_content_change(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path, "alpha", "first description")
        cache = tmp_path / "cache"
        backend1 = CountingBackend()
        await SkillIndex([skill], backend1, cache_dir=cache).build()

        # Edit the SKILL.md — content hash changes → rebuild.
        skill.file_path.write_text(
            "---\ndescription: second description\n---\n# alpha", encoding="utf-8"
        )
        changed = Skill(
            name="alpha",
            description="second description",
            file_path=skill.file_path,
            base_dir=skill.base_dir,
            source=str(tmp_path),
        )
        backend2 = CountingBackend()
        await SkillIndex([changed], backend2, cache_dir=cache).build()
        assert backend2.embed_calls == 1

    @pytest.mark.asyncio
    async def test_cache_miss_when_skill_set_changes(self, tmp_path: Path) -> None:
        a = _make_skill(tmp_path, "alpha", "first")
        cache = tmp_path / "cache"
        backend1 = CountingBackend()
        await SkillIndex([a], backend1, cache_dir=cache).build()

        b = _make_skill(tmp_path, "beta", "second")
        backend2 = CountingBackend()
        await SkillIndex([a, b], backend2, cache_dir=cache).build()
        assert backend2.embed_calls == 1  # different hash, rebuilt

    @pytest.mark.asyncio
    async def test_meta_records_identity_fields(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        backend = CountingBackend(dim=16)
        await SkillIndex(skills, backend, cache_dir=cache).build()
        (meta_path,) = cache.glob("*.skills.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["dim"] == 16
        assert meta["count"] == 1
        assert meta["backend"] == "CountingBackend"
        assert meta["order"] == ["alpha"]  # exact name order is part of identity
        assert isinstance(meta["hash"], str) and len(meta["hash"]) == 64

    @pytest.mark.asyncio
    async def test_empty_skill_list(self, tmp_path: Path) -> None:
        backend = CountingBackend()
        index = SkillIndex([], backend, cache_dir=tmp_path / "cache")
        await index.build()
        assert await index.select("anything") == []
        assert backend.embed_calls == 0


class TestSelect:
    @pytest.mark.asyncio
    async def test_selects_relevant_skill_above_threshold(self, tmp_path: Path) -> None:
        pytest_skill = _make_skill(
            tmp_path,
            "pytest-helper",
            "Use when writing or running pytest unit tests in Python projects.",
        )
        banana_skill = _make_skill(
            tmp_path, "banana-bread", "A recipe for baking moist banana bread at home."
        )
        index = SkillIndex(
            [pytest_skill, banana_skill], LocalEmbedder(), cache_dir=tmp_path / "cache"
        )
        matches = await index.select("write pytest unit tests for my python module")
        assert pytest_skill in matches
        assert banana_skill not in matches

    @pytest.mark.asyncio
    async def test_threshold_respected(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path, "pytest-helper", "pytest unit tests in python")
        index = SkillIndex([skill], LocalEmbedder(), cache_dir=tmp_path / "cache")
        assert await index.select("pytest unit tests") != []
        assert await index.select("pytest unit tests", threshold=1.01) == []

    @pytest.mark.asyncio
    async def test_k_caps_results(self, tmp_path: Path) -> None:
        skills = [
            _make_skill(tmp_path, f"skill-{i}", f"deploy kubernetes cluster number {i}")
            for i in range(5)
        ]
        index = SkillIndex(skills, LocalEmbedder(), cache_dir=tmp_path / "cache")
        matches = await index.select("deploy kubernetes cluster", k=2)
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_hidden_skills_excluded_from_selection(self, tmp_path: Path) -> None:
        hidden = _make_skill(
            tmp_path,
            "secret-helper",
            "Use when writing or running pytest unit tests in Python projects.",
            hide=True,
        )
        visible = _make_skill(
            tmp_path, "other-skill", "Unrelated to any query anyone would ever send."
        )
        index = SkillIndex([hidden, visible], LocalEmbedder(), cache_dir=tmp_path / "cache")
        matches = await index.select("write pytest unit tests", threshold=0.0)
        assert hidden not in matches
        assert index.degraded is False  # not a fallback, a filter

    @pytest.mark.asyncio
    async def test_backend_failure_degrades_to_local_embedder(self, tmp_path: Path) -> None:
        # RS-15: a failing primary backend falls back to the offline local
        # embedder — selection still happens, no static listing.
        visible = _make_skill(
            tmp_path,
            "pytest-helper",
            "Use when writing or running pytest unit tests in Python projects.",
        )
        index = SkillIndex([visible], FailingBackend(), cache_dir=tmp_path / "cache")
        matches = await index.select("write pytest unit tests for my python module")
        assert matches == [visible]  # the local fallback did the matching
        assert index.degraded is False
        assert index.backend_failures == 1
        assert any("local embedder" in w for w in index.warnings)

    @pytest.mark.asyncio
    async def test_backend_failure_memoized_across_selects(self, tmp_path: Path) -> None:
        visible = _make_skill(tmp_path, "alpha", "deploy kubernetes clusters")
        index = SkillIndex([visible], FailingBackend(), cache_dir=tmp_path / "cache")
        await index.select("deploy kubernetes cluster")
        await index.select("deploy kubernetes cluster again")
        # One failed primary call total — the second select skips straight
        # to the memoized local fallback instead of retrying the API.
        assert index.backend_failures == 1
        assert index.degraded is False

    @pytest.mark.asyncio
    async def test_full_listing_only_when_local_also_fails(self, tmp_path: Path) -> None:
        visible = _make_skill(tmp_path, "alpha", "first")
        hidden = _make_skill(tmp_path, "beta", "second", hide=True)
        index = SkillIndex([visible, hidden], BrokenLocalEmbedder(), cache_dir=tmp_path / "cache")
        matches = await index.select("anything")
        assert matches == [visible]  # ALL non-hidden fallback
        assert index.degraded is True
        assert any("static listing" in w for w in index.warnings)


class TestRenderBlock:
    def test_exact_format(self) -> None:
        skills = [
            Skill(
                name="alpha",
                description="first skill",
                file_path=Path("/x/alpha/SKILL.md"),
                base_dir=Path("/x/alpha"),
                source="test",
            ),
            Skill(
                name="beta",
                description="second skill",
                file_path=Path("/x/beta/SKILL.md"),
                base_dir=Path("/x/beta"),
                source="test",
            ),
        ]
        assert render_block(skills) == (
            "Skills are specialized knowledge. If one matches your task, you MUST read "
            "`skill://<name>` before proceeding.\n"
            "<skills>\n"
            "- alpha: first skill\n"
            "- beta: second skill\n"
            "</skills>"
        )

    def test_empty_list_returns_empty_string(self) -> None:
        assert render_block([]) == ""


class ModelBackend(CountingBackend):
    """CountingBackend with an API-style ``model``/``base_url`` identity."""

    def __init__(self, model: str = "embed-1", base_url: str = "http://x/v1") -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url


class TestOrderSensitivity:
    """RS-01: the cache is keyed by the ORDERED skill sequence."""

    @pytest.mark.asyncio
    async def test_reordered_skills_miss_and_rebuild(self, tmp_path: Path) -> None:
        a = _make_skill(tmp_path, "alpha", "first skill")
        b = _make_skill(tmp_path, "beta", "second skill")
        cache = tmp_path / "cache"
        backend1 = CountingBackend()
        await SkillIndex([a, b], backend1, cache_dir=cache).build()
        assert backend1.embed_calls == 1

        # Same set, different order → different content hash → rebuild.
        # Loading the old cache would map row 0 onto the wrong skill.
        backend2 = CountingBackend()
        await SkillIndex([b, a], backend2, cache_dir=cache).build()
        assert backend2.embed_calls == 1

    @pytest.mark.asyncio
    async def test_meta_order_mismatch_is_a_miss(self, tmp_path: Path) -> None:
        a = _make_skill(tmp_path, "alpha", "first skill")
        b = _make_skill(tmp_path, "beta", "second skill")
        cache = tmp_path / "cache"
        await SkillIndex([a, b], CountingBackend(), cache_dir=cache).build()
        # Tamper: keep everything valid except the recorded name order.
        (meta_path,) = cache.glob("*.skills.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["order"] = ["beta", "alpha"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        backend = CountingBackend()
        await SkillIndex([a, b], backend, cache_dir=cache).build()
        assert backend.embed_calls == 1  # not loaded despite matching hash


class TestBackendIdentity:
    """RS-05: embedding model + base_url are part of the cache identity."""

    @pytest.mark.asyncio
    async def test_same_model_hits_cache(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, ModelBackend(model="embed-1"), cache_dir=cache).build()
        backend = ModelBackend(model="embed-1")
        await SkillIndex(skills, backend, cache_dir=cache).build()
        assert backend.embed_calls == 0

    @pytest.mark.asyncio
    async def test_different_model_same_dim_rebuilds(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, ModelBackend(model="embed-1"), cache_dir=cache).build()
        # Equal dim, different model: must NOT reuse the other model's vectors.
        backend = ModelBackend(model="embed-2")
        await SkillIndex(skills, backend, cache_dir=cache).build()
        assert backend.embed_calls == 1

    @pytest.mark.asyncio
    async def test_different_base_url_rebuilds(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, ModelBackend(base_url="http://a/v1"), cache_dir=cache).build()
        backend = ModelBackend(base_url="http://b/v1")
        await SkillIndex(skills, backend, cache_dir=cache).build()
        assert backend.embed_calls == 1


class TestCacheCorruption:
    """RS-06 + RS-12: corrupt/mismatched caches rebuild, never crash."""

    @pytest.mark.asyncio
    async def test_npz_embeds_content_hash(self, tmp_path: Path) -> None:
        import numpy as np

        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, CountingBackend(), cache_dir=cache).build()
        with np.load(_cache_npz(cache)) as data:
            assert (
                str(data["content_hash"])
                == json.loads(next(cache.glob("*.skills.meta.json")).read_text(encoding="utf-8"))[
                    "hash"
                ]
            )

    @pytest.mark.asyncio
    async def test_npz_hash_mismatch_rebuilds(self, tmp_path: Path) -> None:
        import numpy as np

        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, CountingBackend(), cache_dir=cache).build()
        # Rewrite the npz with a foreign hash — simulates an interleaved
        # writer whose meta still describes a different payload.
        npz = _cache_npz(cache)
        with np.load(npz) as data:
            vectors = data["vectors"]
        np.savez_compressed(npz, vectors=vectors, content_hash=np.array("deadbeef"))

        backend = CountingBackend()
        await SkillIndex(skills, backend, cache_dir=cache).build()
        assert backend.embed_calls == 1

    @pytest.mark.asyncio
    async def test_truncated_npz_rebuilds(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, CountingBackend(), cache_dir=cache).build()
        npz = _cache_npz(cache)
        blob = npz.read_bytes()
        npz.write_bytes(blob[: len(blob) // 2])  # truncate mid-archive

        backend = CountingBackend()
        await SkillIndex(skills, backend, cache_dir=cache).build()
        assert backend.embed_calls == 1  # rebuilt, did not crash

    @pytest.mark.asyncio
    async def test_wrong_backend_class_rebuilds(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, ModelBackend(), cache_dir=cache).build()
        # Different class name (no model attr) → meta identity mismatch.
        backend = CountingBackend()
        await SkillIndex(skills, backend, cache_dir=cache).build()
        assert backend.embed_calls == 1

    @pytest.mark.asyncio
    async def test_wrong_dim_rebuilds(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, CountingBackend(dim=8), cache_dir=cache).build()
        backend = CountingBackend(dim=16)
        await SkillIndex(skills, backend, cache_dir=cache).build()
        assert backend.embed_calls == 1

    @pytest.mark.asyncio
    async def test_garbage_meta_rebuilds(self, tmp_path: Path) -> None:
        skills = [_make_skill(tmp_path, "alpha", "first skill")]
        cache = tmp_path / "cache"
        await SkillIndex(skills, CountingBackend(), cache_dir=cache).build()
        next(cache.glob("*.skills.meta.json")).write_text("{not json", encoding="utf-8")
        backend = CountingBackend()
        await SkillIndex(skills, backend, cache_dir=cache).build()
        assert backend.embed_calls == 1


class TestSelectDeterminism:
    @pytest.mark.asyncio
    async def test_picked_sorted_by_discovery_key(self, tmp_path: Path) -> None:
        # RS-13: score order must not leak into the rendered block — two
        # turns matching the SAME set must render byte-identically.
        zeta = _make_skill(tmp_path, "zeta", "deploy kubernetes cluster")
        alpha = _make_skill(tmp_path, "alpha", "deploy kubernetes cluster")
        index = SkillIndex([zeta, alpha], LocalEmbedder(), cache_dir=tmp_path / "cache")
        matches = await index.select("deploy kubernetes cluster", threshold=0.0)
        assert [skill.name for skill in matches] == ["alpha", "zeta"]


class TestCacheKeyedByRoots:
    @pytest.mark.asyncio
    async def test_distinct_roots_get_distinct_cache_files(self, tmp_path: Path) -> None:
        # RS-17: alternating projects must not evict each other's cache.
        root_a = tmp_path / "project-a"
        root_b = tmp_path / "project-b"
        skill_a = _make_skill(root_a, "alpha", "first skill")
        skill_b = _make_skill(root_b, "beta", "second skill")
        cache = tmp_path / "cache"
        await SkillIndex([skill_a], CountingBackend(), cache_dir=cache).build()
        await SkillIndex([skill_b], CountingBackend(), cache_dir=cache).build()
        assert len(list(cache.glob("*.skills.npz"))) == 2

        # Each project still hits its own cache after the other wrote.
        backend = CountingBackend()
        await SkillIndex([skill_a], backend, cache_dir=cache).build()
        assert backend.embed_calls == 0

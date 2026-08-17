"""Skill-routing upgrade tests: globs force-include, hybrid scoring, and
the compaction-keyed re-freeze of the session knowledge block.

The four behaviors here share one theme — selection must be right at the
edges, not just the happy path: an author's path claim (globs) outranks the
embedder, a lexically-exact description must survive a semantic miss, an
imported ecosystem skill must never shadow a native one, and a compaction
must re-open a frozen selection exactly once.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator import session_factory
from local_operator.skills.api import default_skill_roots
from local_operator.skills.discovery import Skill, discover_skills
from local_operator.skills.embeddings import LocalEmbedder
from local_operator.skills.index import SkillIndex, _hybrid_scores

_SKILLS_SUBDIR = Path(".local-operator") / "skills"


def _make_skill(
    root: Path,
    dirname: str,
    description: str,
    *,
    globs: tuple[str, ...] = (),
    hide: bool = False,
) -> Skill:
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\ndescription: {description}\n---\n# {dirname}")
    return Skill(
        name=dirname,
        description=description,
        file_path=skill_md,
        base_dir=skill_dir,
        source=str(root),
        globs=globs,
        hide=hide,
    )


def _write_skill_md(
    root: Path,
    dirname: str,
    description: str,
    body_marker: str,
) -> None:
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {dirname}\ndescription: {description}\n---\n\n{body_marker}"
    )


def _bare_skill(name: str, description: str) -> Skill:
    """A Skill pointing at a fake path — for pure scoring-function tests
    that never stat or embed (no disk writes, no cache identity)."""
    fake = Path("/nonexistent") / name
    return Skill(
        name=name,
        description=description,
        file_path=fake,
        base_dir=fake.parent,
        source=str(fake.parent),
    )


class TestGlobsForceInclude:
    """Globs match the session cwd (or a suffix of it) and file-path-like
    query tokens; matches are forced in at 1.0, bypassing the cosine
    threshold but still capped by k and the hidden filter."""

    @pytest.mark.asyncio
    async def test_glob_matches_cwd_suffix(self, tmp_path: Path) -> None:
        skill = _make_skill(
            tmp_path,
            "backend-helper",
            "A recipe for moist banana bread.",  # semantically unrelated
            globs=("src/**",),
        )
        index = SkillIndex([skill], LocalEmbedder(), cache_dir=tmp_path / "cache")
        matches = await index.select("something entirely unrelated", cwd=Path("/x/repo/src/app"))
        assert matches == [skill]

    @pytest.mark.asyncio
    async def test_glob_matches_query_path_token(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path, "terraform", "Baking guides.", globs=("*.tf",))
        index = SkillIndex([skill], LocalEmbedder(), cache_dir=tmp_path / "cache")
        matches = await index.select("please review main.tf when you can")
        assert matches == [skill]

    @pytest.mark.asyncio
    async def test_glob_bypasses_threshold_but_not_k(self, tmp_path: Path) -> None:
        skills = [
            _make_skill(
                tmp_path,
                f"path-skill-{i}",
                "Completely unrelated filler description.",
                globs=("*.tf",),
            )
            for i in range(10)
        ]
        index = SkillIndex(skills, LocalEmbedder(), cache_dir=tmp_path / "cache")
        matches = await index.select("review main.tf", threshold=0.99)
        assert len(matches) == 8  # k=8 default still caps the force-include

    @pytest.mark.asyncio
    async def test_hidden_glob_skill_still_excluded(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path, "secret", "Hidden helper.", globs=("*.tf",), hide=True)
        index = SkillIndex([skill], LocalEmbedder(), cache_dir=tmp_path / "cache")
        assert await index.select("review main.tf", threshold=0.99) == []

    @pytest.mark.asyncio
    async def test_no_globs_no_force_include(self, tmp_path: Path) -> None:
        # The boost must come from the globs field, not from any path-ish
        # query token alone.
        skill = _make_skill(tmp_path, "plain", "Unrelated description.")
        index = SkillIndex([skill], LocalEmbedder(), cache_dir=tmp_path / "cache")
        assert await index.select("review main.tf", threshold=0.99) == []

    @pytest.mark.asyncio
    async def test_glob_outranks_semantic_match_in_ranking(self, tmp_path: Path) -> None:
        # Both admitted (threshold 0.0); the glob hit must not LOSE to a
        # stronger cosine match in the k ordering.
        semantic = _make_skill(tmp_path, "deploy-docs", "deploy kubernetes cluster to prod east")
        globbed = _make_skill(tmp_path, "aa-path-skill", "Banana bread.", globs=("*.tf",))
        index = SkillIndex([semantic, globbed], LocalEmbedder(), cache_dir=tmp_path / "cache")
        matches = await index.select(
            "deploy kubernetes cluster; also review main.tf", k=1, threshold=0.0
        )
        assert matches == [globbed]

    def test_formula_directly(self) -> None:
        # Name "alpha" makes the indexed text "alpha: alpha beta gamma delta"
        # — the SAME word set as the query, so Jaccard is exactly 1.0.
        exact = _bare_skill("alpha", "alpha beta gamma delta")
        disjoint = _bare_skill("other", "one two three four")
        scores = _hybrid_scores("alpha beta gamma delta", [exact, disjoint], [0.10, 0.10], None)
        assert scores[0] == pytest.approx(0.6 * 0.10 + 0.4 * 1.0)
        assert scores[1] == pytest.approx(0.10)  # max() keeps cosine for jac=0

    @pytest.mark.asyncio
    async def test_lexically_exact_survives_semantic_miss(self, tmp_path: Path) -> None:
        class MissingBackend:
            """A semantic embedder that misses everything: the query vector
            is orthogonal to every corpus row, so cosine is 0.0 for all
            skills. Distinguished by call arity — ``select`` always embeds
            exactly one query against an already-built multi-row matrix, so
            the corpus must hold two skills for the split to be unambiguous."""

            dim = 2
            default_threshold = 0.3

            async def embed(self, texts: list[str]) -> list[list[float]]:
                if len(texts) == 1:
                    return [[0.0, 1.0]]  # the query
                return [[1.0, 0.0] for _ in texts]  # the corpus rows

        exact = _make_skill(tmp_path, "deploy", "deploy the kubernetes cluster to prod east")
        filler = _make_skill(tmp_path, "filler", "unrelated words entirely so")
        index = SkillIndex([exact, filler], MissingBackend(), cache_dir=tmp_path / "cache")
        matches = await index.select("deploy the kubernetes cluster to prod east")
        assert matches == [exact]  # cosine 0.0, Jaccard 1.0 → 0.4 > 0.3

    @pytest.mark.asyncio
    async def test_pure_noise_stays_below_threshold(self, tmp_path: Path) -> None:
        class MissingBackend:
            dim = 2
            default_threshold = 0.3

            async def embed(self, texts: list[str]) -> list[list[float]]:
                if len(texts) == 1:
                    return [[0.0, 1.0]]
                return [[1.0, 0.0] for _ in texts]

        skill = _make_skill(tmp_path, "deploy", "deploy the kubernetes cluster to prod east")
        filler = _make_skill(tmp_path, "filler", "unrelated words entirely so")
        index = SkillIndex([skill, filler], MissingBackend(), cache_dir=tmp_path / "cache")
        assert await index.select("how do I bake sourdough at altitude") == []

    @pytest.mark.asyncio
    async def test_off_corpus_noise_with_local_embedder(self, tmp_path: Path) -> None:
        # Same contract through the REAL embedder: unrelated skills must not
        # leak in just because the blend raises every score.
        deploy = _make_skill(tmp_path, "deploy", "deploy kubernetes cluster to prod east")
        bread = _make_skill(tmp_path, "banana-bread", "baking moist banana bread at home")
        index = SkillIndex([deploy, bread], LocalEmbedder(), cache_dir=tmp_path / "cache")
        assert await index.select("quantum entanglement podcast recommendations") == []


class _CountingIndex:
    """Minimal index double: records every query it is asked to score."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def select(self, query: str, **kwargs: object) -> list[Skill]:
        self.queries.append(query)
        return []


class TestCompactionRefreeze:
    """The knowledge block freezes after the first query — unless the
    transcript's latest compaction entry id changed, which re-opens
    selection exactly once (the head rewrite already killed the cache)."""

    @pytest.mark.asyncio
    async def test_same_compaction_id_stays_frozen(self) -> None:
        index = _CountingIndex()
        hooks = session_factory._KnowledgeHooks(index=index)  # type: ignore[arg-type]
        first = await session_factory._select_knowledge_block(hooks, "first task")
        second = await session_factory._select_knowledge_block(hooks, "later task")
        assert first == second
        assert index.queries == ["first task"]

    @pytest.mark.asyncio
    async def test_new_compaction_id_reselects_once(self) -> None:
        index = _CountingIndex()
        hooks = session_factory._KnowledgeHooks(index=index)  # type: ignore[arg-type]
        await session_factory._select_knowledge_block(hooks, "first task")
        refrozen = await session_factory._select_knowledge_block(
            hooks, "first task\nwe compacted everything", compaction_id="c1"
        )
        assert refrozen is not None
        again = await session_factory._select_knowledge_block(
            hooks, "first task\nwe compacted everything", compaction_id="c1"
        )
        assert index.queries == ["first task", "first task\nwe compacted everything"]
        assert hooks.frozen_compaction_id == "c1"
        assert again == refrozen  # frozen again under the new id

    def test_latest_compaction_id(self) -> None:
        def transcript(entries: list[SimpleNamespace]) -> SimpleNamespace:
            return SimpleNamespace(entries=lambda: list(entries))

        user = dict(type="message", payload={"role": "user", "content": [{"text": "hi"}]})
        assert (
            session_factory._latest_compaction_id(transcript([SimpleNamespace(id="m1", **user)]))
            is None
        )
        assert (
            session_factory._latest_compaction_id(
                transcript(
                    [
                        SimpleNamespace(id="m1", **user),
                        SimpleNamespace(id="c1", type="compaction", payload={}),
                        SimpleNamespace(id="c2", type="compaction", payload={}),
                    ]
                )
            )
            == "c2"
        )  # newest wins, matching replay semantics
        broken = SimpleNamespace(entries=lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert session_factory._latest_compaction_id(broken) is None

    @pytest.mark.asyncio
    async def test_provider_reselects_after_compaction_lands(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import local_operator.prompts_api as prompts_api

        monkeypatch.setattr(
            prompts_api, "build_system_blocks", lambda *args, **kwargs: ["head", "tail"]
        )

        class FakeTranscript:
            def __init__(self) -> None:
                self._entries: list[SimpleNamespace] = [
                    SimpleNamespace(
                        id="m1",
                        type="message",
                        payload={"role": "user", "content": [{"text": "deploy stuff"}]},
                    )
                ]

            def entries(self) -> list[SimpleNamespace]:
                return list(self._entries)

        transcript = FakeTranscript()
        index = _CountingIndex()
        hooks = session_factory._KnowledgeHooks(index=index)  # type: ignore[arg-type]
        provider = session_factory._make_system_blocks_provider([], transcript, hooks)

        await provider()
        await provider()  # no compaction yet: still frozen
        assert index.queries == ["deploy stuff"]

        transcript._entries.append(
            SimpleNamespace(
                id="c9",
                type="compaction",
                payload={"summary": "we were deploying kubernetes"},
            )
        )
        await provider()
        assert len(index.queries) == 2  # compaction re-opened selection
        assert "we were deploying kubernetes" in index.queries[1]
        assert "deploy stuff" in index.queries[1]  # latest user query + summary

        await provider()  # same compaction id: frozen again
        assert len(index.queries) == 2


class TestEcosystemRootsAndCollision:
    """Ecosystem roots scan only what exists, yield to native roots on name
    collisions, and are replaced wholesale by the env override."""

    @pytest.mark.asyncio
    async def test_native_root_wins_collision_and_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".claude" / "skills").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))

        project = tmp_path / "repo"
        native_root = project / _SKILLS_SUBDIR
        _write_skill_md(native_root, "deploy", "native deploy helper", "native body")
        _write_skill_md(
            home / ".claude" / "skills", "deploy", "claude deploy helper", "claude body"
        )

        skills, warnings = discover_skills(default_skill_roots(project))
        deployed = [skill for skill in skills if skill.name == "deploy"]
        assert len(deployed) == 1
        assert deployed[0].description == "native deploy helper"
        assert any("conflict" in warning for warning in warnings)

        # Env override replaces the ecosystem set wholesale: pointing it at
        # an empty dir drops the claude skill even though ~/.claude exists.
        extra = tmp_path / "extra-skills"
        extra.mkdir()
        monkeypatch.setenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", str(extra))
        skills, warnings = discover_skills(default_skill_roots(project))
        assert [skill.name for skill in skills] == ["deploy"]
        assert deployed[0].description == "native deploy helper"
        assert not any("conflict" in warning for warning in warnings)

    def test_missing_ecosystem_roots_are_not_scanned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".claude" / "skills").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))

        roots = default_skill_roots(tmp_path / "repo")
        assert home / ".claude" / "skills" in roots
        assert home / ".codex" / "skills" not in roots
        assert home / ".omp" / "agent" / "skills" not in roots
        # Native home root still precedes every ecosystem root.
        assert roots.index(home / _SKILLS_SUBDIR) < roots.index(home / ".claude" / "skills")

    def test_empty_env_disables_ecosystem_roots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".claude" / "skills").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", "")

        roots = default_skill_roots(tmp_path / "repo")
        assert home / ".claude" / "skills" not in roots

    def test_env_override_replaces_default_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        mine = tmp_path / "my-skills"
        mine.mkdir()
        (home / ".claude" / "skills").mkdir(parents=True)
        monkeypatch.setenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", f"{mine}:{tmp_path / 'nope'}")

        roots = default_skill_roots(tmp_path / "repo")
        assert mine in roots
        assert home / ".claude" / "skills" not in roots  # replaced, not extended
        assert tmp_path / "nope" not in roots  # nonexistent entries dropped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

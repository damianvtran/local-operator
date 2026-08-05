"""LocalEmbedder calibration: the contract behind the shipped threshold.

RS-03: the previous 0.18 threshold sat BELOW the measured noise floor of its
own vector space (median unrelated-pair cosine 0.31 at dim 512), so with
k=8 selection degenerated toward full listing — exactly what blows the
≤30k start-of-session token budget (docs/REWRITE.md performance contract).

This module ships the recalibration as a TEST: an 8-skill corpus spanning
unrelated domains, absolute (not relative) assertions on the separation,
and the pinned shipped constants (dim 4096, threshold 0.27 = midpoint of the
measured gap between the worst unrelated score and the best match score).
The test is the contract — the constants in ``embeddings.py`` exist because
this corpus produces them.
"""

from __future__ import annotations

from statistics import median

import pytest

from local_operator.skills.embeddings import LocalEmbedder

#: 8 realistic skill descriptions across unrelated domains — the shape the
#: index actually embeds ("<name>: <description>").
CORPUS: dict[str, str] = {
    "minerva-observability": (
        "Datadog observability playbooks for production logs, traces, "
        "metrics, monitors, dashboards, and incident investigations."
    ),
    "calendar-support": (
        "Google Workspace calendar support: scheduling events, checking "
        "free/busy windows, and managing meeting invites."
    ),
    "coding-conventions": (
        "Coding conventions and engineering standards for this repository: "
        "formatting, linting, naming, and the code review checklist."
    ),
    "cooking-recipes": (
        "Cooking recipes and kitchen techniques: baking bread, roasting "
        "vegetables, pan sauces, and seasonal meal ideas."
    ),
    "travel-planning": (
        "Travel planning for trips: booking flights and hotels, building "
        "day-by-day itineraries, and visa requirements."
    ),
    "legal-drafting": (
        "Legal drafting and contract review: clause analysis, liability "
        "terms, and regulatory compliance documentation."
    ),
    "image-generation": (
        "Image generation with diffusion models: prompt engineering, aspect "
        "ratios, and visual asset creation."
    ),
    "data-analysis": (
        "Data analysis workflows with pandas and SQL: cleaning datasets, "
        "aggregation, visualization, and reporting."
    ),
}

#: A query that clearly matches one skill, paired with that skill's name.
CALIBRATION_QUERY = (
    "what linting and formatting rules should i follow for code review in this repo",
    "coding-conventions",
)

# Contract bars from the RS-03 fix (measured: median ≈ 0.07, match ≈ 0.42,
# worst unrelated ≈ 0.11 — the bars leave wide margin on every side).
MAX_UNRELATED_PAIR_MEDIAN = 0.10
MIN_MATCH_SCORE = 0.25
MAX_UNRELATED_QUERY_SCORE = 0.15


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class TestCalibration:
    def test_shipped_dim_is_4096(self) -> None:
        assert LocalEmbedder().dim == 4096

    def test_unrelated_pair_median_below_bar(self) -> None:
        """All 28 unrelated skill-vs-skill pairs cluster well below the
        threshold — the median must leave headroom, not just dip under."""
        embedder = LocalEmbedder()
        vecs = [embedder.embed_one(f"{name}: {desc}") for name, desc in CORPUS.items()]
        pairs = [_cos(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        assert len(pairs) == 28
        assert median(pairs) < MAX_UNRELATED_PAIR_MEDIAN, (
            f"median unrelated cosine {median(pairs):.4f} >= "
            f"{MAX_UNRELATED_PAIR_MEDIAN}: noise floor too high, threshold is "
            "not filtering"
        )

    def test_matching_query_separates_from_unrelated(self) -> None:
        """The calibration query's top skill clears 0.25 while every
        unrelated skill stays under 0.15 — a real gap, not an ordering."""
        embedder = LocalEmbedder()
        query_text, expected_name = CALIBRATION_QUERY
        query_vec = embedder.embed_one(query_text)
        scores = {
            name: _cos(query_vec, embedder.embed_one(f"{name}: {desc}"))
            for name, desc in CORPUS.items()
        }
        match_score = scores[expected_name]
        unrelated_max = max(score for name, score in scores.items() if name != expected_name)
        assert match_score >= MIN_MATCH_SCORE, f"match score {match_score:.4f} < {MIN_MATCH_SCORE}"
        assert unrelated_max < MAX_UNRELATED_QUERY_SCORE, (
            f"max unrelated score {unrelated_max:.4f} >= " f"{MAX_UNRELATED_QUERY_SCORE}"
        )

    def test_shipped_threshold_is_the_measured_gap_midpoint(self) -> None:
        """``default_threshold`` is the shipped constant; it must sit inside
        the measured separation gap of the calibration corpus."""
        embedder = LocalEmbedder()
        query_text, expected_name = CALIBRATION_QUERY
        query_vec = embedder.embed_one(query_text)
        scores = {
            name: _cos(query_vec, embedder.embed_one(f"{name}: {desc}"))
            for name, desc in CORPUS.items()
        }
        match_score = scores[expected_name]
        unrelated_max = max(score for name, score in scores.items() if name != expected_name)
        threshold = embedder.default_threshold
        assert unrelated_max < threshold < match_score, (
            f"threshold {threshold} outside gap ({unrelated_max:.4f}, " f"{match_score:.4f})"
        )

    @pytest.mark.asyncio
    async def test_multi_skill_separation_at_shipped_threshold(self) -> None:
        """RS-23: end-to-end through SkillIndex at the SHIPPED threshold —
        for each calibration-style query, the matching skill is selected and
        every unrelated skill is excluded, with absolute score assertions."""
        from pathlib import Path

        from local_operator.skills.discovery import Skill
        from local_operator.skills.index import SkillIndex

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills = []
            for name, desc in CORPUS.items():
                skill_dir = root / name
                skill_dir.mkdir()
                skill_md = skill_dir / "SKILL.md"
                skill_md.write_text(f"---\ndescription: {desc}\n---\n# {name}")
                skills.append(
                    Skill(
                        name=name,
                        description=desc,
                        file_path=skill_md,
                        base_dir=skill_dir,
                        source=str(root),
                    )
                )
            index = SkillIndex(skills, LocalEmbedder(), cache_dir=root / "cache")
            await index.build()

            embedder = LocalEmbedder()
            cases = [
                (CALIBRATION_QUERY[0], CALIBRATION_QUERY[1]),
                (
                    "debug the production latency spike using datadog logs, "
                    "traces, and dashboards",
                    "minerva-observability",
                ),
                (
                    "plan a five day trip to japan with hotels and an itinerary",
                    "travel-planning",
                ),
            ]
            for query_text, expected_name in cases:
                matches = await index.select(query_text)
                names = {skill.name for skill in matches}
                assert (
                    expected_name in names
                ), f"query {query_text!r} did not select {expected_name}"
                assert len(matches) == 1, f"query {query_text!r} selected unrelated skills: {names}"
                # Absolute score shape: match clears the bar, unrelated don't.
                query_vec = embedder.embed_one(query_text)
                score_map = {
                    skill.name: _cos(
                        query_vec, embedder.embed_one(f"{skill.name}: {skill.description}")
                    )
                    for skill in skills
                }
                assert score_map[expected_name] >= MIN_MATCH_SCORE
                unrelated = [score for name, score in score_map.items() if name != expected_name]
                assert max(unrelated) < MAX_UNRELATED_QUERY_SCORE
                assert embedder.default_threshold > max(unrelated)

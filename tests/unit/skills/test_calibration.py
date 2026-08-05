"""LocalEmbedder calibration: the contract behind the shipped threshold.

History, because this constant has moved twice and both moves had a reason:

* 0.18 -> 0.27 (RS-03). At **dim 512** the median unrelated-pair cosine was
  0.31, so 0.18 sat BELOW the noise floor and k=8 selection degenerated
  toward full listing — exactly what blows the <=30k start-of-session token
  budget (docs/REWRITE.md performance contract).
* 0.27 -> 0.18 (this module). The dim is now **4096**, where the unrelated
  noise floor is 0.07-0.08 and the best unrelated score across a real
  15-skill corpus is 0.150. The old 0.27 was additionally derived from
  keyword-rich "clearly matching" queries scoring >=0.42; measured against
  the short phrasings people actually type it returned NOTHING for 44% of
  queries whose correct skill still ranked FIRST. Selecting nothing is the
  expensive failure mode: the agent proceeds without the playbook and no
  warning fires, because zero matches is a legal result.

Do not raise this back toward 0.27 on the strength of the RS-03 note alone —
that measurement belongs to a vector space with 8x fewer dimensions. If the
dim changes again, re-measure both ends of the gap and move the midpoint.

This module ships the recalibration as a TEST: an 8-skill corpus spanning
unrelated domains for the separation assertions, a realistic short-query set
for the recall assertion, and the pinned shipped constants (dim 4096,
threshold 0.18 = midpoint of the measured gap). The test is the contract —
the constants in ``embeddings.py`` exist because these corpora produce them.
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


#: The six skills the realistic queries route to, with descriptions trimmed
#: from the real shipped skills. Kept separate from CORPUS: that one is an
#: 8-way unrelated-domain spread for SEPARATION, this one is a realistic
#: same-organisation set where several entries share vocabulary ("tenant",
#: "prod", "minerva") — which is exactly the condition a threshold has to
#: survive, and the condition an 8-way spread of unrelated domains hides.
REALISTIC_CORPUS: dict[str, str] = {
    "minerva-platform-deployments": (
        "End-to-end runbook for deploying, promoting, and rolling back Minerva "
        "services and interfaces on EKS, including shipping an MR to qa, prod, "
        "or prod-2, Kubernetes manifests, ingress, Terraform, ECR, and ArgoCD "
        "promotion, rollback, and deploy pipelines."
    ),
    "minerva-support-workspace": (
        "Minerva support workspace workflow for Slack, Gmail, Google Calendar, "
        "and Drive. Support triage, customer issue context gathering, Slack "
        "thread summaries or replies, Gmail support requests, and composing "
        "customer support notes."
    ),
    "minerva-usage-metrics": (
        "Minerva usage and data-audit playbooks. Customer search usage, search "
        "volume, tenant activity, ClickHouse metrics tables, search counts, "
        "token and cost usage, and reporting exact usage windows for "
        "production or QA."
    ),
    "minerva-observability": (
        "Datadog observability playbooks for support and incident "
        "investigations. Production or QA logs, traces, spans, metrics, "
        "monitors, error searches, and latency bottlenecks."
    ),
    "minerva-software-development": (
        "Minerva SDLC and engineering guardrails for implementing, reviewing, "
        "testing, and releasing code changes, plus architecture, MR prep, and "
        "the agent review gate every merge request needs."
    ),
    "minerva-admin-apis": (
        "Minerva operational API playbooks for support and tenant "
        "administration. Admin API user and organization management, tenant "
        "limits and entitlements, user invites, and LaunchDarkly feature flags."
    ),
}


#: Short, realistic phrasings — the shape a user actually types, NOT the
#: keyword-stuffed ideal query the original 0.27 was derived from. Each pairs
#: a query with the skill that must be selected for it.
REALISTIC_QUERIES: list[tuple[str, str]] = [
    ("deploy this MR to qa", "minerva-platform-deployments"),
    ("roll back prod-2", "minerva-platform-deployments"),
    ("summarize the slack thread and reply", "minerva-support-workspace"),
    ("customer search usage last month", "minerva-usage-metrics"),
    ("check production logs for errors", "minerva-observability"),
    ("open an MR and get it reviewed", "minerva-software-development"),
]

#: Queries with no matching skill in the corpus. The threshold must reject
#: every one of them — recall is worthless if it is bought with noise.
OFF_CORPUS_QUERIES: list[str] = [
    "translate this poem into french",
    "what is the capital of peru",
    "sort this list of integers in python",
    "explain quantum entanglement",
    "aaaaaa bbbbbb cccccc",
]


class TestRealisticQueryRecall:
    """The defect the 0.18 recalibration fixes.

    At 0.27 every one of these queries ranked its correct skill FIRST and was
    still dropped, because a short query's cosine lands in the 0.21-0.29 band.
    Ranking was never the problem; the cut was.
    """

    def _corpus_scores(self, embedder: LocalEmbedder, query: str) -> dict[str, float]:
        query_vec = embedder.embed_one(query)
        return {
            name: _cos(query_vec, embedder.embed_one(f"{name}: {desc}"))
            for name, desc in REALISTIC_CORPUS.items()
        }

    def test_shipped_threshold_admits_every_realistic_query(self) -> None:
        embedder = LocalEmbedder()
        missed: list[str] = []
        for query, expected in REALISTIC_QUERIES:
            scores = self._corpus_scores(embedder, query)
            if scores[expected] < embedder.default_threshold:
                missed.append(f"{query!r} -> {expected} scored {scores[expected]:.4f}")
        assert not missed, "threshold drops relevant skills:\n  " + "\n  ".join(missed)

    def test_correct_skill_ranks_first_for_realistic_queries(self) -> None:
        """Separate from the threshold: if ranking breaks, no threshold saves
        it, so the two failure modes get separate tests."""
        embedder = LocalEmbedder()
        wrong: list[str] = []
        for query, expected in REALISTIC_QUERIES:
            scores = self._corpus_scores(embedder, query)
            top = max(scores, key=lambda name: scores[name])
            if top != expected:
                wrong.append(f"{query!r} -> {top} (expected {expected})")
        assert not wrong, "ranking regressed:\n  " + "\n  ".join(wrong)

    def test_off_corpus_queries_select_nothing(self) -> None:
        """The other half of the contract: 100% recall must cost 0% noise."""
        embedder = LocalEmbedder()
        leaked: list[str] = []
        for query in OFF_CORPUS_QUERIES:
            scores = self._corpus_scores(embedder, query)
            best = max(scores, key=lambda name: scores[name])
            if scores[best] >= embedder.default_threshold:
                leaked.append(f"{query!r} -> {best} scored {scores[best]:.4f}")
        assert not leaked, "threshold admits unrelated skills:\n  " + "\n  ".join(leaked)

    def test_threshold_sits_inside_the_realistic_gap(self) -> None:
        """Pins the MARGIN, not just the constant: the shipped threshold must
        stay strictly between the worst true match and the best false one, so
        a future embedder change that narrows the gap fails here loudly."""
        embedder = LocalEmbedder()
        worst_relevant = min(self._corpus_scores(embedder, q)[exp] for q, exp in REALISTIC_QUERIES)
        best_unrelated = max(
            max(self._corpus_scores(embedder, q).values()) for q in OFF_CORPUS_QUERIES
        )
        assert best_unrelated < embedder.default_threshold < worst_relevant, (
            f"threshold {embedder.default_threshold} outside realistic gap "
            f"({best_unrelated:.4f}, {worst_relevant:.4f})"
        )

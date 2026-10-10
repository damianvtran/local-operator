"""The builtin skill catalog: integrity, discovery order, resolution, routing.

The catalog ships inside the package (``skills/api.py``'s
:data:`PACKAGED_SKILL_ROOT`) and is appended LAST to the default root list, so a
user copy in any earlier root always wins collisions. These tests pin the
shipped contracts: the content budgets, the discovery precedence (and its
collision warning), reads through the closed ``skill://`` resolver, one routing
query per skill through the real LocalEmbedder selection, and the
classification roster's state budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.classification.context import (
    DEFAULT_MAX_STATE_CHARS,
    Candidate,
    build_state,
    serialized_size,
)
from local_operator.skills.api import (
    PACKAGED_SKILL_ROOT,
    default_skill_roots,
    make_skill_resolver,
)
from local_operator.skills.discovery import discover_skills, parse_frontmatter
from local_operator.skills.embeddings import LocalEmbedder
from local_operator.skills.index import SkillIndex

#: The shipped catalog, in discovery order (``(name.lower(), name, path)``;
#: every name is lowercase kebab-case, so this is plain alphabetical).
EXPECTED_NAMES = (
    "browser-qa",
    "code-review",
    "data-analysis",
    "design-qa",
    "engineering-principles",
    "financial-analysis",
    "frontend-design",
    "responding-to-review",
    "scheduling",
    "skill-authoring",
    "systematic-debugging",
    "test-driven-development",
    "verification-before-completion",
    "writing",
)

#: The content spec's budgets: the description is the routing signal that rides
#: every semantic listing, so it is capped per skill and in total; the body is
#: read on demand and should stay a procedure. Measured at the shipped content:
#: descriptions max 220 / total 2958, bodies max 52 lines / 2827 bytes.
_DESCRIPTION_MAX_CHARS = 220
_DESCRIPTIONS_TOTAL_MAX_CHARS = 3500
_BODY_MAX_LINES = 120
_BODY_MAX_BYTES = 6 * 1024
_REFERENCES_MAX = 3

#: One representative query per shipped skill, taken from the routing battery
#: the content pass was validated against: all 32 battery rows pass against
#: this exact catalog under the offline LocalEmbedder at its shipped default
#: threshold (0.19). A row the crude hashed-n-gram router cannot see is
#: deliberately not asserted — the guides smoke's precedent
#: (tests/unit/guides/test_guides.py).
_ROUTING_ROWS = (
    ("verify this change actually works before I report it done", "verification-before-completion"),
    ("write a failing test first then fix the bug", "test-driven-development"),
    ("debug why this test is flaky and find the root cause", "systematic-debugging"),
    ("refactor this with the smallest possible change", "engineering-principles"),
    ("review this pull request diff for bugs", "code-review"),
    ("respond to the review comments on my PR with fixes", "responding-to-review"),
    ("check this UI for contrast and spacing problems", "design-qa"),
    ("design a landing page that doesn't look generic", "frontend-design"),
    ("test this web app in a browser and check console errors", "browser-qa"),
    ("analyse this CSV dataset and check for missing values", "data-analysis"),
    ("reconcile these financial statements for the quarter", "financial-analysis"),
    ("schedule a meeting across time zones", "scheduling"),
    ("write a new skill for our repo", "skill-authoring"),
    ("rewrite this copy so it doesn't sound like AI", "writing"),
)


def _skill_dirs() -> list[Path]:
    return sorted(
        (child for child in PACKAGED_SKILL_ROOT.iterdir() if child.is_dir()),
        key=lambda path: path.name,
    )


def _frontmatter_and_body(text: str) -> tuple[str, str]:
    """Split a SKILL.md into its YAML block and the markdown body after it."""
    parts = text.split("---", 2)
    assert len(parts) == 3 and not parts[0].strip(), "unexpected SKILL.md shape"
    return parts[1], parts[2]


@pytest.fixture
def synthetic_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch HOME with no ecosystem roots, so root order is deterministic.

    ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` is scrubbed explicitly: the suite
    allow-lists it (it is a read-only directory list), and a cell asserting
    exact roots cannot tolerate an inherited value it never wrote.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", raising=False)
    return home


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_names_match_directories_and_discovery_agrees() -> None:
    directories = _skill_dirs()
    assert [directory.name for directory in directories] == list(EXPECTED_NAMES)

    for directory in directories:
        metadata = parse_frontmatter(
            (directory / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        )
        assert metadata.get("name") == directory.name

    skills, warnings = discover_skills([PACKAGED_SKILL_ROOT])
    assert not warnings
    assert [skill.name for skill in skills] == list(EXPECTED_NAMES)
    assert all(skill.resource_type == "skill" for skill in skills)


def test_descriptions_stay_within_the_content_budget() -> None:
    skills, _ = discover_skills([PACKAGED_SKILL_ROOT])

    assert all(skill.description.strip() for skill in skills)
    assert max(len(skill.description) for skill in skills) <= _DESCRIPTION_MAX_CHARS
    assert sum(len(skill.description) for skill in skills) <= _DESCRIPTIONS_TOTAL_MAX_CHARS


def test_bodies_and_references_stay_within_the_content_budget() -> None:
    for directory in _skill_dirs():
        _, body = _frontmatter_and_body(
            (directory / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        )
        assert len(body.splitlines()) <= _BODY_MAX_LINES, directory.name
        assert len(body.encode("utf-8")) <= _BODY_MAX_BYTES, directory.name

        references = sorted((directory / "references").glob("*.md"))
        assert len(references) <= _REFERENCES_MAX, directory.name


def test_skill_trees_contain_only_skill_md_and_references() -> None:
    """No scripts, no stray files: Phase 1 ships markdown only."""
    for directory in _skill_dirs():
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory)
            assert relative.parts[0] in (
                "SKILL.md",
                "references",
            ), f"{directory.name}: unexpected file {relative}"
            assert path.suffix == ".md", f"{directory.name}: non-markdown file {relative}"


# ---------------------------------------------------------------------------
# Discovery order and collisions
# ---------------------------------------------------------------------------


def test_packaged_root_is_last_in_the_default_roots(synthetic_home: Path, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    roots = default_skill_roots(project)

    assert roots[-1] == PACKAGED_SKILL_ROOT
    # No ecosystem dirs exist under the scratch home, so the home root sits
    # directly before the packaged catalog.
    assert roots[-2] == synthetic_home / ".local-operator" / "skills"


def test_all_builtins_are_discovered_through_the_default_roots(
    synthetic_home: Path, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    skills, warnings = discover_skills(default_skill_roots(project))

    assert not warnings
    builtin = [skill for skill in skills if skill.file_path.is_relative_to(PACKAGED_SKILL_ROOT)]
    assert [skill.name for skill in builtin] == list(EXPECTED_NAMES)
    assert all(skill.resource_type == "skill" for skill in builtin)


def test_a_user_copy_in_an_earlier_root_wins_and_warns(
    synthetic_home: Path, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    copy_dir = project / ".local-operator" / "skills" / "writing"
    copy_dir.mkdir(parents=True)
    (copy_dir / "SKILL.md").write_text(
        "---\nname: writing\ndescription: Project writing helper\n---\n# project writing",
        encoding="utf-8",
    )

    skills, warnings = discover_skills(default_skill_roots(project))

    writing = [skill for skill in skills if skill.name == "writing"]
    assert len(writing) == 1
    assert writing[0].file_path == copy_dir / "SKILL.md"

    shadow_warnings = [warning for warning in warnings if "'writing'" in warning]
    assert len(shadow_warnings) == 1
    assert str(PACKAGED_SKILL_ROOT / "writing" / "SKILL.md") in shadow_warnings[0]
    assert str(copy_dir / "SKILL.md") in shadow_warnings[0]
    assert "earlier root wins" in shadow_warnings[0]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_builtin_bodies_and_references_resolve_through_the_resolver() -> None:
    skills, _ = discover_skills([PACKAGED_SKILL_ROOT])
    resolver = make_skill_resolver({skill.name: skill for skill in skills})

    body = resolver("skill://verification-before-completion")
    assert body is not None
    assert "No claim ships without evidence" in body

    reference = resolver("skill://design-qa/references/checks-web.md")
    assert reference is not None
    assert "Web checks" in reference


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected"), _ROUTING_ROWS)
async def test_each_builtin_routes_from_a_representative_task(
    tmp_path: Path, query: str, expected: str
) -> None:
    skills, _ = discover_skills([PACKAGED_SKILL_ROOT])
    index = SkillIndex(skills, LocalEmbedder(), cache_dir=tmp_path / "cache")
    await index.build()

    selected = await index.select(query)

    assert expected in {skill.name for skill in selected}


# ---------------------------------------------------------------------------
# Classification roster budget
# ---------------------------------------------------------------------------


def test_a_builtin_only_roster_fits_the_classification_state_budget() -> None:
    """The catalog alone must not push a rung-0 classification state over the cap.

    Deterministic and backend-free: the roster is built from the discovered
    descriptions and measured with the same serializer the request uses. The
    long-description assertion pins the RUNG: had the state needed rung 2, every
    line over 120 chars would travel trimmed (with a ``…`` marker) and the full
    text could not appear.
    """
    skills, _ = discover_skills([PACKAGED_SKILL_ROOT])
    candidates: list[Candidate] = [
        Candidate(
            kind="skill",
            name=skill.name,
            description=skill.description,
            resource_url=f"skill://{skill.name}",
        )
        for skill in skills
    ]

    state = build_state(user_message="help me with this task", context=None, candidates=candidates)

    assert serialized_size(state) <= DEFAULT_MAX_STATE_CHARS
    serialized = json.dumps(state, ensure_ascii=False)
    assert any(len(skill.description) > 120 and skill.description in serialized for skill in skills)

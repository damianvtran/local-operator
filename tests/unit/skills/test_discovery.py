"""Discovery tests: frontmatter rules, scan shape, collision + ordering."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.skills.discovery import (
    Skill,
    discover_skills,
    parse_frontmatter,
    scan_skills_dir,
)


def _write_skill(
    root: Path,
    dirname: str,
    *,
    name: str | None = None,
    description: str | None = "A test skill.",
    enabled: bool | None = None,
    hide: bool | None = None,
    disable_model_invocation: bool | None = None,
    body: str = "# Body",
) -> Path:
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True)
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not None:
        lines.append(f"description: {description}")
    if enabled is not None:
        lines.append(f"enabled: {str(enabled).lower()}")
    if hide is not None:
        lines.append(f"hide: {str(hide).lower()}")
    if disable_model_invocation is not None:
        lines.append(f"disable-model-invocation: {str(disable_model_invocation).lower()}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("\n".join(lines), encoding="utf-8")
    return skill_md


class TestParseFrontmatter:
    def test_parses_all_recognized_keys(self) -> None:
        text = (
            "---\nname: foo\ndescription: Does things\nenabled: true\n"
            "hide: true\ndisable-model-invocation: true\n---\nbody"
        )
        meta = parse_frontmatter(text)
        assert meta["name"] == "foo"
        assert meta["description"] == "Does things"
        assert meta["enabled"] is True
        assert meta["hide"] is True
        assert meta["disable-model-invocation"] is True

    def test_no_frontmatter_returns_empty(self) -> None:
        assert parse_frontmatter("# Just markdown") == {}

    def test_unterminated_block_returns_empty(self) -> None:
        assert parse_frontmatter("---\nname: foo\nno closing fence") == {}

    def test_malformed_yaml_returns_empty(self) -> None:
        assert parse_frontmatter("---\nname: [unclosed\n---\nbody") == {}

    def test_non_dict_yaml_returns_empty(self) -> None:
        assert parse_frontmatter("---\n- a\n- b\n---\nbody") == {}


class TestScanSkillsDir:
    def test_frontmatter_rules(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "with-name", name="renamed")
        _write_skill(root, "fallback-name")  # no name -> dir name
        _write_skill(root, "disabled", enabled=False)
        _write_skill(root, "no-desc", description=None)
        _write_skill(root, "blank-desc", description="   ")
        _write_skill(root, "hidden", hide=True)
        _write_skill(root, "dmi", disable_model_invocation=True)

        skills = {s.name: s for s in scan_skills_dir(root, source="test")}

        assert "renamed" in skills  # frontmatter name wins
        assert skills["renamed"].base_dir == root / "with-name"
        assert "fallback-name" in skills  # name fell back to dir name
        assert "disabled" not in skills
        assert "no-desc" not in skills
        assert "blank-desc" not in skills
        assert skills["hidden"].hide is True
        assert skills["dmi"].hide is True
        assert all(s.source == "test" for s in skills.values())

    def test_skips_dotdirs_and_nondirs_and_is_non_recursive(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "real")
        _write_skill(root, ".hidden-dir")
        # non-recursive: nested skill under a skill dir is not picked up
        nested = root / "real" / "nested"
        nested.mkdir()
        (nested / "SKILL.md").write_text("---\ndescription: nested\n---\n", encoding="utf-8")
        # a bare file named SKILL.md directly under root is ignored (no include_self)
        (root / "SKILL.md").write_text("---\ndescription: self\n---\n", encoding="utf-8")

        skills = scan_skills_dir(root, source="test")
        assert [s.name for s in skills] == ["real"]

    def test_include_self_picks_up_root_skill_md(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        (root / "SKILL.md").write_text("---\ndescription: self\n---\n", encoding="utf-8")
        skills = scan_skills_dir(root, source="test", include_self=True)
        assert [s.name for s in skills] == ["skills"]  # falls back to root dir name

    def test_realpath_dedupe_shared_seen(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "target")
        link = tmp_path / "linked-skills" / "alias"
        link.parent.mkdir()
        (link).symlink_to(root / "target")

        seen: set[str] = set()
        first = scan_skills_dir(root, source="a", seen=seen)
        second = scan_skills_dir(link.parent, source="b", seen=seen)
        assert [s.name for s in first] == ["target"]
        assert second == []  # same physical SKILL.md, not reloaded

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert scan_skills_dir(tmp_path / "nope", source="x") == []


class TestDiscoverSkills:
    def test_deterministic_order_ci_then_exact_then_path(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "a", name="zeta")
        _write_skill(root, "b", name="Alpha")
        _write_skill(root, "c", name="alpha")
        _write_skill(root, "d", name="beta")

        skills, warnings = discover_skills([root])
        assert warnings == []
        # "Alpha" < "alpha" on the exact-name tiebreak (uppercase first)
        assert [s.name for s in skills] == ["Alpha", "alpha", "beta", "zeta"]

    def test_earlier_root_wins_collision_with_warning(self, tmp_path: Path) -> None:
        root1 = tmp_path / "project"
        root2 = tmp_path / "home"
        _write_skill(root1, "dup", description="project version")
        _write_skill(root2, "dup", description="home version")
        _write_skill(root2, "only-home")

        skills, warnings = discover_skills([root1, root2])
        by_name = {s.name: s for s in skills}
        assert by_name["dup"].description == "project version"
        assert "only-home" in by_name
        assert len(warnings) == 1
        assert "dup" in warnings[0]
        assert "shadowed" in warnings[0]

    def test_missing_roots_skipped_silently(self, tmp_path: Path) -> None:
        root = tmp_path / "real"
        _write_skill(root, "a")
        skills, warnings = discover_skills([tmp_path / "ghost", root])
        assert [s.name for s in skills] == ["a"]
        assert warnings == []

    def test_realpath_dedupe_across_roots_no_warning(self, tmp_path: Path) -> None:
        root1 = tmp_path / "r1"
        _write_skill(root1, "shared")
        root2 = tmp_path / "r2"
        root2.mkdir()
        (root2 / "shared-link").symlink_to(root1 / "shared")

        skills, warnings = discover_skills([root1, root2])
        assert [s.name for s in skills] == ["shared"]
        assert warnings == []  # dedupe is not a collision

    def test_skill_model_fields(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        path = _write_skill(root, "mine", description="Does stuff")
        skills, _ = discover_skills([root])
        skill: Skill = skills[0]
        assert skill.file_path == path
        assert skill.base_dir == root / "mine"
        assert skill.source == str(root)
        assert skill.hide is False

    def test_enabled_false_string_not_dropped(self, tmp_path: Path) -> None:
        # Only boolean false drops; a weird string value is kept (yaml parses
        # it as a truthy non-False value), matching omp's `enabled === false`.
        root = tmp_path / "skills"
        (root / "odd").mkdir(parents=True)
        (root / "odd" / "SKILL.md").write_text(
            "---\ndescription: odd\nenabled: 'false'\n---\n", encoding="utf-8"
        )
        skills, _ = discover_skills([root])
        assert [s.name for s in skills] == ["odd"]

    def test_stable_across_repeated_scans(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        for letter in ["c", "a", "b"]:
            _write_skill(root, letter)
        first, _ = discover_skills([root])
        second, _ = discover_skills([root])
        assert [s.name for s in first] == [s.name for s in second]
        assert [str(s.file_path) for s in first] == [str(s.file_path) for s in second]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

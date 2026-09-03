"""api.py tests: default_skill_roots walk-up + dedupe (RS-04/RS-11) and the
make_skill_resolver adapter (RS-09)."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.skills.api import default_skill_roots, make_skill_resolver
from local_operator.skills.discovery import Skill
from local_operator.skills.protocol import MAX_READ_BYTES

_SKILLS_SUBDIR = Path(".local-operator") / "skills"


def _make_skill(root: Path, name: str) -> Skill:
    base_dir = root / name
    base_dir.mkdir(parents=True)
    skill_md = base_dir / "SKILL.md"
    skill_md.write_text(f"---\ndescription: {name} does things\n---\n# {name}")
    return Skill(
        name=name,
        description=f"{name} does things",
        file_path=skill_md,
        base_dir=base_dir,
        source=str(root),
    )


class TestDefaultSkillRoots:
    def test_walk_up_collects_ancestor_roots_deepest_first(self, tmp_path: Path) -> None:
        project = tmp_path / "repo" / "nested"
        project.mkdir(parents=True)
        roots = default_skill_roots(project)
        assert roots[0] == project / _SKILLS_SUBDIR
        assert (tmp_path / "repo" / _SKILLS_SUBDIR) in roots
        assert (tmp_path / _SKILLS_SUBDIR) in roots
        # Deepest (most project-local) root comes first.
        assert roots.index(project / _SKILLS_SUBDIR) < roots.index(tmp_path / _SKILLS_SUBDIR)

    def test_home_root_appended_last(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        roots = default_skill_roots(tmp_path / "elsewhere")
        assert roots[-1] == home / _SKILLS_SUBDIR

    def test_cwd_outside_home_still_walks_up(self, tmp_path: Path) -> None:
        # RS-04/RS-11: a repo at /opt-style paths (outside $HOME) must still
        # get its project-local roots — the walk goes to the filesystem root,
        # not just to home.
        repo = tmp_path / "srv" / "app"
        repo.mkdir(parents=True)
        roots = default_skill_roots(repo)
        assert (repo / _SKILLS_SUBDIR) in roots
        assert (tmp_path / "srv" / _SKILLS_SUBDIR) in roots

    def test_home_inside_walk_is_not_duplicated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / "repo").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        roots = default_skill_roots(home / "repo")
        # The walk reaches home; the appended home root dedupes against it.
        assert roots.count(home / _SKILLS_SUBDIR) == 1

    def test_symlinked_ancestor_deduped_by_realpath(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        monkeypatch.setenv("HOME", str(tmp_path / "nothome"))
        roots = default_skill_roots(link / "project")
        # /link/project walks through real/project (via realpath); the same
        # physical directory appears once, not twice.
        keys = {str(r.resolve()) for r in roots if r.exists()}
        assert len(keys) == len([r for r in roots if r.exists()])

    def test_default_cwd_is_process_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        roots = default_skill_roots()
        assert roots[0] == tmp_path / _SKILLS_SUBDIR


class TestMakeSkillResolver:
    def test_non_skill_url_returns_none(self, tmp_path: Path) -> None:
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")})
        assert resolver("file:///etc/passwd") is None
        assert resolver("http://example.com") is None
        assert resolver("memory://notes") is None

    def test_skill_url_returns_content(self, tmp_path: Path) -> None:
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")})
        content = resolver("skill://alpha")
        assert content is not None
        assert "# alpha" in content

    def test_empty_name_returns_error_message_with_available_skills(self, tmp_path: Path) -> None:
        skills = {
            "alpha": _make_skill(tmp_path, "alpha"),
            "beta": _make_skill(tmp_path, "beta"),
        }
        resolver = make_skill_resolver(skills)
        content = resolver("skill://")
        assert content is not None
        assert "missing a name" in content
        assert "Available skills: alpha, beta" in content

    def test_unknown_name_returns_error_message_as_content(self, tmp_path: Path) -> None:
        # RS-09: the adapter never raises; the available-names list reaches
        # the model as a clean tool result instead of an exception envelope.
        skills = {
            "alpha": _make_skill(tmp_path, "alpha"),
            "beta": _make_skill(tmp_path, "beta"),
        }
        resolver = make_skill_resolver(skills)
        content = resolver("skill://gamma")
        assert content is not None
        assert "Unknown skill: gamma" in content
        assert "Available: alpha, beta" in content

    def test_unsafe_path_returns_error_message_as_content(self, tmp_path: Path) -> None:
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")})
        content = resolver("skill://alpha/../../etc/passwd")
        assert content is not None
        assert "not allowed" in content

    def test_error_content_fits_in_one_tool_result(self, tmp_path: Path) -> None:
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")})
        content = resolver("skill://gamma")
        assert content is not None
        assert len(content.encode("utf-8")) < MAX_READ_BYTES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

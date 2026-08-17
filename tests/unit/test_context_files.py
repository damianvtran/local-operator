"""Repository guidance (AGENTS.md/CLAUDE.md) discovery and injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.context_files import (
    discover_context_files,
    load_repo_guidance,
    render_context_files,
)


def _make_repo(root: Path, with_git: bool = True) -> Path:
    repo = root / "proj"
    repo.mkdir(parents=True)
    if with_git:
        (repo / ".git").mkdir()
    return repo


def test_discovers_ancestors_farthest_first(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text("root guidance\n")
    (repo / "src").mkdir()
    (repo / "src" / "AGENTS.md").write_text("src guidance\n")
    files = discover_context_files(repo / "src")
    assert [f.name for f in files] == ["AGENTS.md", "AGENTS.md"]
    assert files[0].parent.name == "proj"
    assert files[1].parent.name == "src"
    rendered = render_context_files(files, repo / "src")
    # Farthest first: the root block precedes the src block.
    assert rendered.index("root guidance") < rendered.index("src guidance")
    assert rendered.startswith("## Repository guidance")
    assert "conversation still wins" in rendered


def test_claude_md_stands_in_when_agents_md_absent(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "CLAUDE.md").write_text("claude guidance\n")
    files = discover_context_files(repo)
    assert len(files) == 1
    assert files[0].name == "CLAUDE.md"
    # Both present: AGENTS.md wins, never both.
    (repo / "AGENTS.md").write_text("agents guidance\n")
    files = discover_context_files(repo)
    assert [f.name for f in files] == ["AGENTS.md"]


def test_stops_at_git_root(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text("in repo\n")
    outside = repo / "above"
    outside.mkdir()
    (outside / "AGENTS.md").write_text("outside repo\n")
    # A cwd INSIDE the repo: the repo root is the boundary.
    inner = repo / "pkg"
    inner.mkdir()
    files = discover_context_files(inner)
    assert len(files) == 1
    assert files[0].read_text() == "in repo\n"


def test_no_git_root_stops_at_home(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, with_git=False)
    (repo / "AGENTS.md").write_text("no git\n")
    files = discover_context_files(repo)
    # tmp_path is outside $HOME and has no git root: the walk stops at the
    # filesystem root boundary without error, finding only the repo's file.
    assert [f.read_text() for f in files] == ["no git\n"]


def test_nearest_cap_and_dedupe(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    nested = repo / "a" / "b" / "c" / "d" / "e" / "f"
    nested.mkdir(parents=True)
    shared = "same bytes everywhere\n"
    for part in ("a", "a/b", "a/b/c", "a/b/c/d", "a/b/c/d/e", "a/b/c/d/e/f"):
        (repo / part / "AGENTS.md").write_text(shared)
    files = discover_context_files(nested)
    # Byte-identical files collapse to one regardless of depth.
    assert len(files) == 1


def test_env_kill_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text("hidden\n")
    monkeypatch.setenv("LOCAL_OPERATOR_CONTEXT_FILES", "0")
    assert discover_context_files(repo) == []
    assert load_repo_guidance(repo) == ""


def test_truncates_oversized_file(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text("x" * (64 * 1024 + 100))
    rendered = load_repo_guidance(repo)
    assert "truncated at 64KiB" in rendered


def test_empty_when_no_files(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert discover_context_files(repo) == []
    assert load_repo_guidance(repo) == ""

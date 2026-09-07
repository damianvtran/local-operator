"""Repository guidance (AGENTS.md/CLAUDE.md) discovery and injection."""

from __future__ import annotations

import re
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


def _oversized(head: str = "", tail_sections: int = 3) -> str:
    """A guidance file whose head exceeds GUIDANCE_HEAD_BYTES, with real
    sections after the cut."""
    filler = "\n".join(f"padding line {i}" for i in range(900))
    body = f"# Project\n\n## Head section\n\n{head}\n{filler}\n"
    for index in range(tail_sections):
        body += f"\n## Late section {index}\n\nrule {index} lives here.\n"
    return body


def test_oversized_file_ships_head_plus_section_index(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text(_oversized())
    rendered = load_repo_guidance(repo)

    # The head is resident...
    assert "## Head section" in rendered
    # ...the tail is not...
    assert "rule 2 lives here." not in rendered
    # ...but every late section is NAMED with a line range, which is the whole
    # point: a rule past the cut used to be invisible AND unreachable.
    for index in range(3):
        assert f"Late section {index}" in rendered
    assert re.search(r"- L\d+-\d+: Late section 0", rendered)
    assert "you MUST `read`" in rendered


def test_index_line_ranges_address_the_real_lines(tmp_path: Path) -> None:
    """The emitted ranges must be what ``read`` needs to see that section."""
    repo = _make_repo(tmp_path)
    guidance = repo / "AGENTS.md"
    guidance.write_text(_oversized())
    rendered = load_repo_guidance(repo)
    lines = guidance.read_text().splitlines()

    found = re.findall(r"- L(\d+)-(\d+): (.+)", rendered)
    assert found, "expected an index"
    for raw_start, raw_end, title in found:
        start, end = int(raw_start), int(raw_end)
        span = lines[start - 1 : end]
        assert span, f"empty span for {title}"
        # A section wholly past the head is addressed from its own heading. A
        # section straddling the cut is offered as its REMAINDER instead, so
        # its range starts at body text rather than at the heading the model
        # has already read.
        if span[0].startswith("#"):
            assert span[0].lstrip("#").strip() == title
        # Either way the span stops before the next same-or-higher heading.
        assert not any(line.startswith("## ") for line in span[1:])


def test_file_within_head_size_ships_whole_without_ceremony(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text("# Tiny\n\n## Rules\n\nBe nice.\n")
    rendered = load_repo_guidance(repo)
    assert "Be nice." in rendered
    # No index, no read-the-file imperative: the model can already see it all.
    assert "NOT included above" not in rendered
    assert "you MUST `read`" not in rendered


def test_head_is_cut_on_a_line_boundary_never_mid_sentence(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # Long lines so a byte cut would land mid-line with near-certainty.
    body = "# P\n\n## S\n\n" + "".join(f"{'sentence ' * 20}end-{i}.\n" for i in range(200))
    (repo / "AGENTS.md").write_text(body)
    rendered = load_repo_guidance(repo)
    head = rendered.split("\nThe rest of this file is NOT included")[0]
    resident = [line for line in head.splitlines() if line.startswith("sentence ")]
    assert resident, "expected some body lines resident"
    # Every resident body line is a COMPLETE line from the source.
    source = set(body.splitlines())
    assert all(line in source for line in resident)


def test_index_ignores_headings_inside_fenced_code_blocks(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    filler = "\n".join(f"padding {i}" for i in range(900))
    body = (
        f"# P\n\n## Head\n\n{filler}\n\n## Real section\n\n"
        "```sh\n# not a heading, a shell comment\n## also not a heading\n```\n"
    )
    (repo / "AGENTS.md").write_text(body)
    rendered = load_repo_guidance(repo)
    assert "Real section" in rendered
    assert "a shell comment" not in rendered
    assert "also not a heading" not in rendered


def test_empty_when_no_files(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert discover_context_files(repo) == []
    assert load_repo_guidance(repo) == ""


def test_cap_keeps_nearest_most_specific_files(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    current = repo
    for index in range(7):
        current = current / f"d{index}"
        current.mkdir()
        (current / "AGENTS.md").write_text(f"guidance {index}\n")
    files = discover_context_files(current)
    assert len(files) == 5
    bodies = [file.read_text() for file in files]
    # Prompt order is farthest->nearest, but the retained SET is the nearest 5.
    assert bodies == [f"guidance {i}\n" for i in range(2, 7)]


def test_symlinked_guidance_is_never_discovered_or_rendered(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("DO NOT SEND")
    link = repo / "AGENTS.md"
    link.symlink_to(secret)

    assert discover_context_files(repo) == []
    # Render repeats the no-follow check: a link swapped in after discovery
    # cannot exploit the gap between the two bounded reads.
    assert render_context_files([link], repo) == ""


def test_guidance_cap_bounds_ingestion_before_hash_and_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.context_files import GUIDANCE_HEAD_BYTES

    repo = _make_repo(tmp_path)
    guidance = repo / "AGENTS.md"
    guidance.write_bytes(b"x" * (GUIDANCE_HEAD_BYTES + 10_000_000))

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("unbounded Path read API used")

    # The old implementation used both and allocated the whole file twice.
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    rendered = load_repo_guidance(repo)

    # One enormous line: nothing to cut on, so the head is the byte prefix and
    # the render stays bounded regardless of how large the file is.
    assert rendered.count("x") <= GUIDANCE_HEAD_BYTES
    assert len(rendered.encode()) < GUIDANCE_HEAD_BYTES + 2_000


def test_scan_ceiling_is_disclosed_rather_than_silently_dropping_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this module exists to fix is a SILENT omission, so the one
    remaining bound must announce itself."""
    import local_operator.context_files as module

    monkeypatch.setattr(module, "MAX_SCAN_BYTES", 9_000)
    repo = _make_repo(tmp_path)
    filler = "\n".join(f"padding line {i}" for i in range(4000))
    (repo / "AGENTS.md").write_text(f"# P\n\n## Head\n\n{filler}\n\n## Way past the scan\n\nx\n")
    rendered = load_repo_guidance(repo)
    assert "index scan stopped at" in rendered

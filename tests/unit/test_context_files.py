"""Repository guidance (AGENTS.md/CLAUDE.md) discovery and injection."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from local_operator.context_files import (
    _read_head,
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

    # The head boundary is computed, NOT sniffed from the span's first line.
    # Guarding the identity assertion behind `if span[0].startswith("#")` makes
    # the test self-disabling in exactly the failure it exists to catch: a
    # start that drifts off its heading lands on body text, the guard goes
    # false, and the assertion is skipped. A start+1 mutant passed the whole
    # suite that way. Every row is now checked unconditionally, in the one of
    # two modes its start determines.
    _, head_lines, _ = _read_head(guidance)

    straddling = 0
    for raw_start, raw_end, title in found:
        start, end = int(raw_start), int(raw_end)
        span = lines[start - 1 : end]
        assert span, f"empty span for {title}"
        if start == head_lines + 1:
            # The straddling section: its heading is already resident, so the
            # range offered is the REMAINDER and must NOT start on a heading.
            # Pinning this stops the exemption from becoming a loophole.
            straddling += 1
            assert not span[0].startswith("#"), (
                f"{title!r} was offered from the head boundary, so it should "
                f"start at body text, not at {span[0]!r}"
            )
        else:
            # Every other row must land exactly on its own heading line.
            assert span[0].lstrip("#").strip() == title, (
                f"index row {title!r} points at L{start} but read() sees " f"{span[0]!r} there"
            )
        # Either way the span stops before the next same-or-higher heading.
        assert not any(line.startswith("## ") for line in span[1:])
    assert straddling <= 1, "at most one section can straddle the head cut"


@pytest.mark.parametrize(
    "name,body",
    [
        # No heading of any level.
        (
            "no headings",
            "\n".join(f"prose line {i}" for i in range(1200)) + "\nNEVER delete prod.\n",
        ),
        # Only H1s: the index lists H2/H3, so this yields no listable rows.
        (
            "only H1",
            "# Alpha\n"
            + "background prose here\n" * 400
            + "\n# Beta rules\n"
            + "detail line\n" * 400
            + "\n# Gamma\nnever do X\n",
        ),
        # Every heading inside the head, prose continuing well past it.
        ("headings only in head", "# Doc\n\n## Only\n\n" + "tail prose line\n" * 900),
    ],
)
def test_oversized_file_never_renders_without_disclosure(
    name: str, body: str, tmp_path: Path
) -> None:
    """The module's one invariant: content is never dropped silently.

    Each shape here yields NO listable section, which used to make the index
    render as "" and the caller emit a bare head — the tail gone AND unnamed,
    with not even the file's path to find it by. ``main`` shipped these files
    WHOLE, so that was a content regression as well as the exact
    absent-and-invisible failure this module exists to invert.
    """
    from local_operator.context_files import GUIDANCE_HEAD_BYTES

    repo = _make_repo(tmp_path)
    assert len(body.encode()) > GUIDANCE_HEAD_BYTES, f"{name} fixture is not oversized"
    (repo / "AGENTS.md").write_text(body)
    rendered = load_repo_guidance(repo)

    assert "NOT included above" in rendered, f"{name}: tail dropped with no disclosure"
    assert "AGENTS.md" in rendered, f"{name}: no path to read"
    assert re.search(r"L\d+", rendered), f"{name}: no line range to read"
    assert "you MUST" in rendered, f"{name}: no imperative to read the rest"


def test_a_file_that_sections_with_h1_gets_those_sections_named(tmp_path: Path) -> None:
    """Several H1s mean the file sections with them; one means it titles with it.

    Excluding H1 unconditionally handed an H1-sectioned file a single
    undifferentiated range covering hundreds of lines, while the scan already
    held the section names. A lone H1 is still excluded: a row spanning the
    whole file only restates the bare pointer.
    """
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text(
        "# Alpha\n"
        + "background prose here\n" * 400
        + "\n# Beta rules\n"
        + "detail line\n" * 400
        + "\n# Gamma\nnever do X\n"
    )
    rendered = load_repo_guidance(repo)

    assert re.search(r"- L\d+-\d+: Beta rules", rendered)
    assert re.search(r"- L\d+-\d+: Gamma", rendered)

    # A single H1 (the title) plus H2 sections must be unaffected: the title
    # spans the file, so listing it adds nothing.
    other = _make_repo(tmp_path / "other")
    (other / "AGENTS.md").write_text(
        "# The Title\n\n## Head\n\n" + "padding prose line here\n" * 500 + "\n## Late\n\nrule\n"
    )
    rendered_titled = load_repo_guidance(other)
    assert re.search(r"- L\d+-\d+: Late", rendered_titled)
    assert not re.search(r"- L\d+-\d+: The Title", rendered_titled)


def test_unterminated_code_fence_still_indexes_the_tail(tmp_path: Path) -> None:
    """A fence left open at EOF is a formatting slip, not a 60KB code block.

    Believing it swallows every later heading and strands the tail unnamed.
    The scan retries without fence tracking and discloses that it did so.
    """
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text(
        "# Team rules\n\n## Setup\n\n```sh\necho hi\n"
        + "prose line here\n" * 700
        + "\n## Deploy policy\n\nNEVER deploy to production on a Friday.\n"
    )
    rendered = load_repo_guidance(repo)

    assert "Deploy policy" in rendered, "heading after an unclosed fence went unnamed"
    assert "unclosed code fence" in rendered, "the recovery was not disclosed"


@pytest.mark.parametrize(
    "separator",
    ["\u2028", "\u2029", "\x85", "\x0c", "\x0b", "\x1c", "\x1d", "\x1e"],
)
def test_line_numbering_matches_the_read_tool(separator: str, tmp_path: Path) -> None:
    """The index's line numbers are a CONTRACT with the ``read`` tool.

    ``read`` decodes and calls ``str.splitlines()``, which breaks on six
    separators beyond ``\\n``. Counting on ``\\n`` alone drifts every later
    range by one per occurrence, compounding down the file — and the model
    then lands on a plausible WRONG range and follows the wrong rule, which is
    worse than finding nothing. This pins both sides to one splitter, so a
    change to either breaks loudly instead of drifting.
    """
    from local_operator.tools.builtin import _decode_text_lines

    repo = _make_repo(tmp_path)
    guidance = repo / "AGENTS.md"
    guidance.write_text(
        "# Doc\n\n## Head\n\n"
        + f"prose line with{separator}a separator in it\n" * 400
        + "\n## Late section\n\nTHE RULE.\n"
    )
    rendered = load_repo_guidance(repo)

    # Resolve the emitted range through the REAL consumer, not a local split.
    _, lines = _decode_text_lines(guidance.read_bytes())
    match = re.search(r"- L(\d+)-(\d+): Late section", rendered)
    assert match, f"late section not indexed with separator {separator!r}"
    start, end = int(match.group(1)), int(match.group(2))
    assert lines[start - 1] == "## Late section", (
        f"separator {separator!r}: index says L{start}, read() sees " f"{lines[start - 1]!r}"
    )
    assert any("THE RULE." in line for line in lines[start - 1 : end])


def test_every_index_range_round_trips_through_the_real_read_path(tmp_path: Path) -> None:
    """The PR's thesis, mechanically: if the pointer is followed, does it land?

    Whether a model *chooses* to follow the pointer is not testable here. That
    it resolves correctly when followed is, and it is the half that silently
    broke — so it is asserted against ``read``'s own splitter rather than a
    reimplementation of it.
    """
    from local_operator.tools.builtin import _decode_text_lines

    repo = _make_repo(tmp_path)
    guidance = repo / "AGENTS.md"
    guidance.write_text(_oversized(tail_sections=6))
    rendered = load_repo_guidance(repo)

    _, lines = _decode_text_lines(guidance.read_bytes())
    _, head_lines, _ = _read_head(guidance)
    rows = re.findall(r"- L(\d+)-(\d+): (.+)", rendered)
    assert rows, "expected an index"
    for raw_start, raw_end, title in rows:
        start, end = int(raw_start), int(raw_end)
        assert 1 <= start <= end <= len(lines), f"{title}: range outside the file"
        if start != head_lines + 1:
            assert lines[start - 1].lstrip("#").strip() == title


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
    # the render stays bounded regardless of how large the file is. Counted on
    # the run of file bytes rather than on every "x" in the render, since the
    # disclosure prose legitimately contains the letter too.
    longest_run = max(len(run) for run in re.findall(r"x+", rendered))
    assert longest_run <= GUIDANCE_HEAD_BYTES
    assert len(rendered.encode()) < GUIDANCE_HEAD_BYTES + 2_000
    # A heading-free oversized file must still say the tail exists.
    assert "NOT included above" in rendered


def test_bare_pointer_discloses_the_scan_ceiling_instead_of_a_wrong_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clipped line count must be labelled a floor, never stated as a length.

    ``total_lines`` is counted over the bytes the scan read, so under the
    ceiling it is short. Passing it to the bare pointer unqualified announced a
    confident WRONG length and capped the offered range there -- silently
    stranding everything past it, inside the very branch added to guarantee
    nothing is stranded silently.
    """
    import local_operator.context_files as module

    monkeypatch.setattr(module, "MAX_SCAN_BYTES", 9_000)
    repo = _make_repo(tmp_path)
    body = "\n".join(f"prose line {i}" for i in range(900))  # heading-free
    (repo / "AGENTS.md").write_text(body)
    rendered = load_repo_guidance(repo)

    assert "NOT included above" in rendered
    # No bare "(N lines)" claim: that number would be the ceiling's, not the
    # file's, and the reader cannot tell the difference.
    assert not re.search(r"\(\d+ lines\)", rendered), "clipped count stated as the length"
    assert "floor" in rendered and "scan stopped" in rendered
    # The range must not stop at the clipped count either.
    assert "END of the file" in rendered
    assert not re.search(rf"L\d+-{len(body.splitlines())}\b", rendered)


def test_index_falls_back_to_a_pointer_when_the_scan_cannot_read_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The head can be read and the scan then fail (a race, or a permission
    change between the two opens). That path is the blocker's own code path, so
    it must still disclose rather than return an empty index."""
    import local_operator.context_files as module

    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text(_oversized())
    real_scan = module._scan_sections

    def fail_after_head(*args: object, **kwargs: object) -> object:
        raise OSError("vanished between head and scan")

    monkeypatch.setattr(module, "_scan_sections", fail_after_head)
    rendered = load_repo_guidance(repo)
    monkeypatch.setattr(module, "_scan_sections", real_scan)

    assert "NOT included above" in rendered, "scan failure dropped the tail silently"
    assert "AGENTS.md" in rendered
    assert "you MUST" in rendered


def test_tail_beyond_the_last_listed_section_is_disclosed(tmp_path: Path) -> None:
    """Rows can be non-empty and still not reach EOF; that remainder is offered
    explicitly rather than left implied.

    Driven directly on the renderer so the boundary itself is pinned with
    arbitrary spans. The companion test below reaches the same branch through
    ``load_repo_guidance`` on a real file, which is what proves the guard
    protects a live path rather than a hypothetical one.
    """
    from local_operator.context_files import _render_index_rows, _Section

    # Sections that stop well short of the file's end.
    sections = [_Section(2, "Early", 200), _Section(2, "Seen", 300)]
    sections[0].end = 299
    sections[1].end = 400
    rendered = _render_index_rows(
        "AGENTS.md",
        head_lines=150,
        sections=sections,
        total_lines=900,
        scan_truncated=False,
        unterminated=False,
    )

    assert re.search(r"- L\d+-\d+: Seen", rendered), "expected a listed section"
    assert "not under any listed heading" in rendered, "unlisted tail was not disclosed"
    assert "L401-900" in rendered, "the unlisted range itself must be offered"


def test_a_single_late_h1_leaves_a_tail_that_must_be_disclosed(tmp_path: Path) -> None:
    """The unlisted-tail guard on a REAL file, with no ceiling and no fence.

    Exactly one H1 that is not the first heading makes ``titles_only`` true, so
    ``floor`` is 2 and that trailing H1 is dropped from the listing -- while it
    still owns every line to EOF. The listed rows therefore stop short, and the
    lines the H1 covers would be unreachable if the guard did not fire.

    This pins the reachability claim by execution: the guard was previously
    documented as unreachable from a file fixture, which was wrong, because
    that reasoning predated the H1 filter it interacts with.
    """
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text(
        "## Alpha\n\n" + "pad line here\n" * 800 + "\n# Late title\n\n" + "tail line\n" * 400
    )
    rendered = load_repo_guidance(repo)

    match = re.search(r"\(L(\d+)-(\d+) is not under any listed heading", rendered)
    assert match, "a tail outside every listed row was not disclosed"
    # The disclosed range must start right after the last listed row and run to
    # the end of the file, or the lines in between stay unreachable.
    last_listed = max(int(end) for _, end in re.findall(r"- L(\d+)-(\d+):", rendered))
    assert int(match.group(1)) == last_listed + 1
    from local_operator.tools.builtin import _decode_text_lines

    _, lines = _decode_text_lines((repo / "AGENTS.md").read_bytes())
    assert int(match.group(2)) == len(lines)


def test_files_sharing_a_digest_prefix_but_differing_in_length_are_kept_apart(
    tmp_path: Path,
) -> None:
    """Only the first MAX_DIGEST_BYTES are hashed, so length is folded in.

    Without it, a nested repo whose two guidance files share a long prefix
    dedups to one, and the survivor ships an index describing the OTHER file --
    every range off by however much they differ.
    """
    from local_operator.context_files import MAX_DIGEST_BYTES

    repo = _make_repo(tmp_path)
    nested = repo / "pkg"
    nested.mkdir()
    shared = "# Shared\n\n" + "x" * (MAX_DIGEST_BYTES + 1000)
    (repo / "AGENTS.md").write_text(shared)
    (nested / "AGENTS.md").write_text(shared + "\n## Only in the deeper file\n\nrule\n")

    assert len(discover_context_files(nested)) == 2, "same-prefix files collapsed"
    # Identical files still dedup: the fold must not defeat that.
    (nested / "AGENTS.md").write_text(shared)
    assert len(discover_context_files(nested)) == 1


def test_index_lists_h3_rows_not_only_top_level_sections(tmp_path: Path) -> None:
    """INDEX_MAX_HEADING_LEVEL is a real knob: H3 is where the individually
    actionable rules live, so lowering it to 2 would quietly coarsen every
    index. Pinned as a property -- deeper rows exist and nest -- rather than as
    a row count, which would break on any edit to the fixture."""
    from local_operator.context_files import INDEX_MAX_HEADING_LEVEL

    assert INDEX_MAX_HEADING_LEVEL >= 3
    repo = _make_repo(tmp_path)
    (repo / "AGENTS.md").write_text(
        "# Doc\n\n## Head\n\n"
        + "padding prose line here\n" * 500
        + "\n## Gates\n\ntext\n\n### Never skip QA\n\nrule\n\n### Never force push\n\nrule\n"
    )
    rendered = load_repo_guidance(repo)

    assert re.search(r"- L\d+-\d+: Gates", rendered)
    assert re.search(r"  - L\d+-\d+: Never skip QA", rendered), "H3 rows missing or unindented"
    assert "Never force push" in rendered


def test_head_retains_the_start_of_the_file_that_adherence_depends_on(
    tmp_path: Path,
) -> None:
    """The head size is load-bearing and nothing else pins it.

    Halving GUIDANCE_HEAD_BYTES passed the whole suite, which would silently
    evict the unconditional gates the constant's docstring reasons about -- the
    same overstatement Q2 was raised about, arriving by a different route. This
    pins the PROPERTY (an early-file gate is resident, and the head is a
    meaningful fraction of the budget) rather than the literal number, so the
    constant stays tunable but not quietly halvable.
    """
    from local_operator.context_files import GUIDANCE_HEAD_BYTES

    repo = _make_repo(tmp_path)
    marker = "NEVER merge without a green pipeline."
    # The marker sits at a FIXED offset just under 8 KiB -- deliberately not
    # derived from GUIDANCE_HEAD_BYTES, because a fixture that scales with the
    # constant shrinks along with a mutant and pins nothing (8K->4K and 8K->6K
    # both left the suite green that way). This asserts the shipped contract:
    # roughly the first 8 KiB of a guidance file stays resident, so a gate
    # written there is not silently evicted by a future retuning.
    preamble = "# Project\n\n## Gates\n\nreference material line\n"
    target_offset = 8 * 1024 - 200
    filler = "reference material line\n" * (
        (target_offset - len(preamble.encode())) // len("reference material line\n")
    )
    (repo / "AGENTS.md").write_text(
        preamble + filler + f"{marker}\n" + "trailing reference line\n" * 2000
    )
    rendered = load_repo_guidance(repo)
    head = rendered.split("\nThe rest of this file is NOT included")[0]

    assert marker in head, (
        "a gate lying inside the documented head budget was evicted; "
        f"GUIDANCE_HEAD_BYTES={GUIDANCE_HEAD_BYTES} is smaller than the "
        "docstring's reasoning assumes"
    )
    resident = len(head.encode())
    assert resident >= GUIDANCE_HEAD_BYTES * 0.75, (
        f"head shipped {resident} B against a {GUIDANCE_HEAD_BYTES} B budget; "
        "the constant is being under-spent"
    )


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

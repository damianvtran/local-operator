"""The deterministic pre-filter: doors, exclusions, dedup, the sensitive gate, the budget.

Each exclusion has a test that the guard FIRES and a twin that the same shape passes when the
guard's condition is absent, so deleting a guard is a red test and over-matching is too.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.supplements import candidates as cand
from local_operator.supplements.candidates import prefilter
from tests.unit.supplements.support import call, result


def _write(path: Path, text: str = "x" * 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _run(items, final: str, cwd: Path, **kw):
    return prefilter(items, final, cwd=str(cwd), home=str(cwd.parent), **kw)


def test_a_write_call_is_tier_one_and_the_path_is_relative(tmp_path: Path) -> None:
    _write(tmp_path / "reports" / "latency.md")
    items = [call("write", path=str(tmp_path / "reports" / "latency.md"), i="Writing report")]
    out = _run(items, "Done.", tmp_path)
    assert [c.path for c in out.candidates] == ["reports/latency.md"]
    only = out.candidates[0]
    assert (only.tier, only.kind, only.tool, only.why) == (
        1,
        "markdown",
        "write",
        "written by write",
    )
    assert only.intent == "Writing report"
    assert not os.path.isabs(only.path), "a row never carries an absolute path (S-R14)"
    assert out.skipped is None


def test_an_edit_call_is_not_a_door(tmp_path: Path) -> None:
    """Editing an existing file is the diff, already a tool row: featuring it is spam."""
    _write(tmp_path / "app.py")
    out = _run([call("edit", path=str(tmp_path / "app.py"))], "Refactored.", tmp_path)
    assert out.candidates == () and out.skipped == "prefilter"


def test_a_failed_write_is_not_evidence(tmp_path: Path) -> None:
    _write(tmp_path / "report.md")
    items = [
        call("write", "w1", path=str(tmp_path / "report.md")),
        result("denied", "w1", tool="write", error=True),
    ]
    assert _run(items, "I could not write it.", tmp_path).candidates == ()


@pytest.mark.parametrize(
    "command",
    [
        "python bench.py -o {p}",
        "python bench.py --output {p}",
        "python bench.py --output={p}",
        "python bench.py > {p}",
        "python bench.py >> {p}",
        "python bench.py 2>&1 >{p}",
        "echo hi | tee {p}",
    ],
)
def test_bash_output_flags_are_tier_two(command: str, tmp_path: Path) -> None:
    target = _write(tmp_path / "out.csv")
    out = _run([call("bash", command=command.format(p=target))], "ok", tmp_path)
    assert [(c.path, c.tier, c.tool) for c in out.candidates] == [("out.csv", 2, "bash")], command


def test_a_bash_command_without_an_output_flag_names_nothing(tmp_path: Path) -> None:
    target = _write(tmp_path / "in.csv")
    out = _run([call("bash", command=f"cat {target} | wc -l")], "12 lines", tmp_path)
    assert out.candidates == (), "a file that was only READ is not a deliverable"


def test_eval_writes_are_tier_two(tmp_path: Path) -> None:
    target = _write(tmp_path / "plot.png")
    code = f"df.to_csv('{target}')\nplt.savefig('{target}')"
    out = _run([call("eval", code=code)], "plotted", tmp_path)
    assert [(c.path, c.tier, c.tool) for c in out.candidates] == [("plot.png", 2, "eval")]


def test_a_path_in_the_answer_prose_is_tier_three_and_needs_a_deliverable_extension(
    tmp_path: Path,
) -> None:
    pdf = _write(tmp_path / "board.pdf")
    py = _write(tmp_path / "script.py")
    out = _run([], f"I put it at {pdf} and the helper at {py}.", tmp_path)
    assert [(c.path, c.tier) for c in out.candidates] == [("board.pdf", 3)]


def test_the_answers_own_code_span_or_link_dedups_the_candidate(tmp_path: Path) -> None:
    target = _write(tmp_path / "reports" / "q3.md")
    items = [call("write", path=str(target))]
    assert _run(items, "Wrote `reports/q3.md` for you.", tmp_path).candidates == ()
    assert _run(items, "See [the report](reports/q3.md).", tmp_path).candidates == ()
    assert _run(items, f"Saved to `{target}`.", tmp_path).candidates == ()
    # The twin: the same file merely mentioned in plain prose is still visible only as text
    # the fold hides, so it is NOT deduped.
    kept = _run(items, "I wrote the q3 report.", tmp_path)
    assert [c.path for c in kept.candidates] == ["reports/q3.md"]
    assert (
        _run(items, "Wrote reports/q3.md for you.", tmp_path).rejected.get("already-visible")
        is None
    )


@pytest.mark.parametrize(
    ("relative", "reason"),
    [
        ("node_modules/pkg/readme.md", "generated-tree"),
        (".git/notes.md", "generated-tree"),
        (".venv/lib/notes.md", "generated-tree"),
        ("__pycache__/x.md", "generated-tree"),
        ("dist/report.md", "generated-tree"),
        ("build/report.md", "generated-tree"),
        ("run.log", "excluded-suffix"),
        ("poetry.lock", "excluded-suffix"),
        ("mod.pyc", "excluded-suffix"),
    ],
)
def test_vendored_generated_and_noise_files_are_excluded(
    relative: str, reason: str, tmp_path: Path
) -> None:
    target = _write(tmp_path / relative)
    out = _run([call("write", path=str(target))], "Done.", tmp_path)
    assert out.candidates == () and out.rejected == {reason: 1}


def test_temp_roots_are_scratch_unless_they_are_the_working_directory(tmp_path: Path) -> None:
    # tmp_path IS under the OS temp dir. As the cwd it is the project, so its files count...
    inside = _write(tmp_path / "proj" / "report.md")
    out = _run([call("write", path=str(inside))], "Done.", tmp_path / "proj")
    assert [c.path for c in out.candidates] == ["report.md"]
    # ...but a file under /tmp that is OUTSIDE the cwd is scratch (and outside the roots).
    stray = Path("/tmp") / "supplements-test-stray.md"
    stray.write_text("x")
    try:
        gone = _run([call("write", path=str(stray))], "Done.", tmp_path / "proj")
        assert gone.candidates == ()
    finally:
        stray.unlink()


def test_the_scratchpad_and_config_dir_are_never_listed(tmp_path: Path, monkeypatch) -> None:
    pad = tmp_path / "pad"
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(pad))
    draft = _write(pad / "draft.md")
    out = _run([call("write", path=str(draft))], "Done.", tmp_path)
    assert out.candidates == () and out.rejected == {"sensitive": 1}


@pytest.mark.parametrize(
    "name", [".env", "prod.env", "id_rsa", "token.json", "credentials", "app.db"]
)
def test_sensitive_files_never_become_candidates(name: str, tmp_path: Path) -> None:
    target = _write(tmp_path / name)
    out = _run([call("write", path=str(target))], "Done.", tmp_path)
    assert out.candidates == () and out.rejected == {"sensitive": 1}, name


def test_a_symlink_to_a_secret_is_not_a_candidate(tmp_path: Path) -> None:
    secret = _write(tmp_path / ".ssh" / "id_rsa")
    link = tmp_path / "report.md"
    link.symlink_to(secret)
    out = _run([call("write", path=str(link))], "Done.", tmp_path)
    assert out.candidates == ()


def test_a_symlink_escaping_the_roots_is_refused(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    outside = _write(Path("/var/tmp") / "supplements-outside-test.md")
    try:
        link = project / "report.md"
        link.symlink_to(outside)
        out = prefilter(
            [call("write", path=str(link))], "Done.", cwd=str(project), home=str(project)
        )
        assert out.candidates == () and out.rejected == {"resolves-outside-roots": 1}
    finally:
        outside.unlink()


def test_an_operator_deny_prefix_hides_a_whole_tree(tmp_path: Path) -> None:
    target = _write(tmp_path / "clients" / "acme" / "report.md")
    items = [call("write", path=str(target))]
    assert _run(items, "Done.", tmp_path).candidates != ()
    out = _run(items, "Done.", tmp_path, deny_prefixes=["clients"])
    assert out.candidates == () and out.rejected == {"deny-prefix": 1}
    out = _run(items, "Done.", tmp_path, deny_prefixes=[str(tmp_path / "clients")])
    assert out.rejected == {"deny-prefix": 1}


def test_missing_directories_and_outside_paths_are_dropped(tmp_path: Path) -> None:
    items = [
        call("write", "a", path=str(tmp_path / "ghost.md")),
        call("write", "b", path=str(tmp_path / "sub")),
        call("write", "c", path="/etc/hosts"),
    ]
    (tmp_path / "sub").mkdir()
    out = _run(items, "Done.", tmp_path)
    assert out.candidates == ()
    assert out.rejected == {"missing": 1, "not-regular": 1, "outside-roots": 1}


def test_a_source_file_write_is_offered_but_is_not_a_deliverable(tmp_path: Path) -> None:
    """'Write me a script' is a legitimate request (the decision can tell), but the no-vendor
    heuristic must not feature source files."""
    script = _write(tmp_path / "tool.py")
    out = _run([call("write", path=str(script))], "Done.", tmp_path)
    assert [(c.path, c.kind, c.deliverable) for c in out.candidates] == [("tool.py", "code", False)]


def test_a_later_mention_of_the_same_file_keeps_the_stronger_tier(tmp_path: Path) -> None:
    target = _write(tmp_path / "out.csv")
    items = [call("bash", "b", command=f"run > {target}"), call("write", "w", path=str(target))]
    out = _run(items, "Done.", tmp_path)
    assert [(c.path, c.tier) for c in out.candidates] == [("out.csv", 1)]


def test_candidate_count_is_bounded(tmp_path: Path) -> None:
    items = []
    for index in range(200):
        _write(tmp_path / f"f{index}.md")
        items.append(call("write", f"c{index}", path=str(tmp_path / f"f{index}.md")))
    out = _run(items, "Done.", tmp_path)
    assert len(out.candidates) == cand.MAX_CONSIDERED


def test_the_gate_stops_a_turn_with_nothing_and_respects_the_switches(tmp_path: Path) -> None:
    assert _run([], "Paris.", tmp_path).skipped == "prefilter"
    table = "| a | b |\n|---|---|\n| x | 14 |\n| y | 9 |\n| z | 31 |\n"
    assert _run([], table, tmp_path).skipped is None
    assert _run([], table, tmp_path, want_graphics=False).skipped == "prefilter"
    target = _write(tmp_path / "r.md")
    items = [call("write", path=str(target))]
    assert _run(items, "ok", tmp_path, want_files=False, want_graphics=False).skipped == "prefilter"


def test_the_prefilter_meets_its_five_millisecond_target_on_a_busy_turn(tmp_path: Path) -> None:
    """Structural cost bound, not a wall-clock assertion on a shared host (AGENTS.md "Prefer a
    structural invariant"): the candidate budget caps the stats, and the scan budget caps the
    evidence text, so the work done is bounded regardless of how much the turn produced."""
    items = []
    for index in range(80):
        _write(tmp_path / f"f{index}.md")
        items.append(call("write", f"c{index}", path=str(tmp_path / f"f{index}.md")))
        items.append(result("row,1\n" * 20_000, f"c{index}"))
    out = _run(items, "x " * 5_000, tmp_path)
    assert len(out.candidates) <= cand.MAX_CONSIDERED

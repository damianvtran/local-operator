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


@pytest.mark.parametrize(
    "name", ["report.md", "report.html", "report.htm", "report.pdf", "report.docx"]
)
def test_deliverable_document_edits_are_metadata_candidates(name: str, tmp_path: Path) -> None:
    target = _write(tmp_path / name)
    out = _run([call("edit", path=str(target))], "Updated document.", tmp_path)
    assert [c.path for c in out.candidates] == [name]
    assert out.candidates[0].why == "updated by edit"


def test_relative_final_deliverable_paths_are_admitted_but_links_still_dedup(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "docs" / "report.md")
    assert [c.path for c in _run([], "Prepared docs/report.md for you.", tmp_path).candidates] == [
        "docs/report.md"
    ]
    assert not _run([], "See [report](docs/report.md).", tmp_path).candidates
    assert not _run([], "See https://example.test/docs/report.md", tmp_path).candidates


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
    # A WRITTEN deliverable the answer links is kept and flagged (recall), not deduped.
    for answer in (
        "Wrote `reports/q3.md` for you.",
        "See [the report](reports/q3.md).",
        f"Saved to `{target}`.",
    ):
        (only,) = _run(items, answer, tmp_path).candidates
        assert (only.path, only.linked_in_answer) == ("reports/q3.md", True), answer
    # ...while a linked file the turn only MENTIONED (tier 3) or a linked script still dedups.
    assert _run([], "See [the report](reports/q3.md).", tmp_path).candidates == ()
    script = _write(tmp_path / "tool.py")
    assert _run([call("write", path=str(script))], "Wrote `tool.py`.", tmp_path).candidates == ()
    # The twin: the same file merely mentioned in plain prose is still visible only as text
    # the fold hides, so it is NOT deduped.
    kept = _run(items, "I wrote the q3 report.", tmp_path)
    assert [(c.path, c.linked_in_answer) for c in kept.candidates] == [("reports/q3.md", False)]
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


# -- bash recall: cd tracking, heredocs, copies, the touched-directory sweep ---------------------
#
# The incident these pin: a turn did ``cd ~/workspace/x; cp A b.jsonl; python3 - <<PY ...
# open('RUN-RECORD.md','w')`` then ``shasum ... > SHA256SUMS`` and ended with no Highlights,
# because the old parser lexed the whole command with ``shlex`` (a python heredoc's quotes
# raise), never tracked the ``cd`` and read ``cp`` as no output at all.


def _bash(tmp_path: Path, command: str, **kw):
    return _run([call("bash", command=command)], "done", tmp_path, **kw)


def test_a_cd_then_redirect_resolves_against_the_cd_directory(tmp_path: Path) -> None:
    _write(tmp_path / "proj" / "out.csv")
    out = _bash(tmp_path, "cd proj && echo a,b > out.csv")
    assert [(c.path, c.tier) for c in out.candidates] == [("proj/out.csv", 2)]


def test_a_home_relative_cd_makes_the_file_display_home_relative_when_cwd_is_elsewhere(
    tmp_path: Path, monkeypatch
) -> None:
    """(c) A project cwd that is NOT home, a deliverable under ``~/workspace``.

    The fixture tree lives under a temp root, which the scratch rule would (rightly) refuse for
    a file outside the cwd; a real home is not scratch, so the rule is lifted for this cell.
    """
    monkeypatch.setattr(cand, "_scratch_roots", lambda: [])
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _write(home / "workspace" / "pkg" / "RUN-RECORD.md")
    command = "cd ~/workspace/pkg; python3 - <<'PY'\nopen('RUN-RECORD.md','w').write('x')\nPY"
    out = prefilter([call("bash", command=command)], "ok", cwd=str(project), home=str(home))
    assert [(c.path, c.kind) for c in out.candidates] == [
        ("~/workspace/pkg/RUN-RECORD.md", "markdown")
    ]
    assert not os.path.isabs(out.candidates[0].path)
    # A file under the cwd keeps the cwd-relative spelling even when the cwd is inside home.
    inner = _write(project / "r.md")
    assert cand._display_path(str(inner), str(project), str(home)) == "r.md"


@pytest.mark.parametrize(
    "command",
    [
        'cd d; cp "$LOCAL_OPERATOR_SCRATCHPAD/led.jsonl" copy.jsonl',
        "cd d; cp a.jsonl copy.jsonl",
        "cd d && mv a.jsonl copy.jsonl",
        "cd d; cat a.jsonl | tee copy.jsonl",
        "cd d; curl -o copy.jsonl https://example.test/x",
        "cd d; python3 - <<'PY'\nimport json\njson.dump({}, open('copy.jsonl', 'w'))\nPY",
        "cd d; python3 - <<'PY'\ns = \"it's\"\nopen('copy.jsonl','w').write(s)\nPY",
        "cd d; python3 -c \"open('copy.jsonl','w').write('x')\"",
        "(cd d; echo x > copy.jsonl)",
    ],
)
def test_bash_destinations_resolve_against_the_cd_directory(command: str, tmp_path: Path) -> None:
    _write(tmp_path / "d" / "copy.jsonl")
    out = _bash(tmp_path, command)
    assert [c.path for c in out.candidates] == ["d/copy.jsonl"], command


def test_a_heredoc_body_that_is_file_text_is_not_scanned_for_python_writes(tmp_path: Path) -> None:
    """``cat > f <<EOF`` bodies are the FILE's content; only an interpreter's stdin is code."""
    _write(tmp_path / "f.md")
    _write(tmp_path / "decoy.md")
    out = _bash(tmp_path, "cat > f.md <<'EOF'\nopen('decoy.md','w')\nEOF")
    assert [c.path for c in out.candidates] == ["f.md"]


def test_an_unknowable_cd_drops_relative_names_instead_of_guessing(tmp_path: Path) -> None:
    _write(tmp_path / "out.csv")  # exists at the session cwd, but the shell was NOT there
    out = _bash(tmp_path, 'cd "$WORK"; echo x > out.csv')
    assert out.candidates == (), "a guess against the session cwd would feature the wrong file"


def test_a_failed_bash_call_wrote_nothing(tmp_path: Path) -> None:
    _write(tmp_path / "out.csv")
    items = [call("bash", "b1", command="echo x > out.csv"), result("boom", "b1", error=True)]
    assert _run(items, "failed", tmp_path).candidates == ()


def test_checksum_manifests_are_text_deliverables_but_other_bare_names_are_not(
    tmp_path: Path,
) -> None:
    assert cand.kind_of("SHA256SUMS") == "text" and cand.kind_of("sha256sums") == "text"
    assert cand.kind_of("pkg.tar.gz.sha256") == "text"
    for name in ("LICENSE", "Makefile", "README", "CHECKS", "Dockerfile"):
        assert cand.kind_of(name) == "other", name
    _write(tmp_path / "SHA256SUMS")
    out = _bash(tmp_path, "shasum -a 256 *.csv > SHA256SUMS")
    assert [(c.path, c.kind, c.deliverable) for c in out.candidates] == [
        ("SHA256SUMS", "text", True)
    ]


def test_the_sweep_finds_files_written_through_a_variable_path_and_only_those(
    tmp_path: Path,
) -> None:
    """``json.dump(d, open(p, 'w'))`` names no literal; the cd directory is listed instead."""
    import time

    work = tmp_path / "w"
    old = _write(work / "old.json")
    os.utime(old, (1.0, 1.0))
    _write(work / "fresh.json")
    _write(work / ".hidden.json")
    _write(work / "run.log")
    _write(work / "notes.py")
    _write(work / "sub" / "deep.json")
    (work / "adir.json").mkdir()
    command = "cd w; python3 - <<'PY'\np = pick()\nopen(p, 'w').write('x')\nPY"
    out = _bash(tmp_path, command, since=time.time() - 60)
    assert [c.path for c in out.candidates] == ["w/fresh.json"]
    # No turn start recorded -> the sweep is OFF (listing "everything" would over-match).
    assert _bash(tmp_path, command).candidates == ()


def test_a_command_with_only_named_outputs_does_not_arm_the_sweep(tmp_path: Path) -> None:
    import time

    _write(tmp_path / "w" / "named.md")
    _write(tmp_path / "w" / "bystander.md")
    out = _bash(tmp_path, "cd w; echo x > named.md", since=time.time() - 60)
    assert [c.path for c in out.candidates] == ["w/named.md"]


def test_the_sweep_is_bounded_in_entries_and_files(tmp_path: Path) -> None:
    import time

    work = tmp_path / "w"
    for index in range(cand.SWEEP_MAX_ENTRIES + 50):
        _write(work / f"f{index:03d}.json", "x")
    command = "cd w; python3 - <<'PY'\nopen(p, 'w').write('x')\nPY"
    out = _bash(tmp_path, command, since=time.time() - 60)
    assert 0 < len(out.candidates) <= cand.SWEEP_MAX_FILES


def test_recall_keeps_every_privacy_rule(tmp_path: Path, monkeypatch) -> None:
    """Negative controls: the new doors cannot admit what the old ones refused."""
    import time

    scratch = tmp_path / "scratchpad"
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(scratch))
    outside = tmp_path / "outside"
    cwd = tmp_path / "proj"
    home = tmp_path / "proj"  # cwd == home: the "outside" tree is under neither
    _write(cwd / "w" / ".env", "TOKEN=1")
    _write(cwd / "w" / "debug.log")
    _write(cwd / "w" / "credentials.json")
    _write(outside / "stolen.csv")
    _write(scratch / "ledger.jsonl")
    command = (
        f"cd w; echo a > .env; echo b > debug.log; echo c > credentials.json;"
        f" cp x {outside}/stolen.csv; cp x {scratch}/ledger.jsonl;"
        " python3 - <<'PY'\nopen(p, 'w').write('x')\nPY"
    )
    out = prefilter(
        [call("bash", command=command)], "ok", cwd=str(cwd), home=str(home), since=time.time() - 60
    )
    assert out.candidates == (), [c.path for c in out.candidates]
    assert out.rejected.get("outside-roots", 0) >= 1


def test_a_linked_name_without_a_path_still_flags_the_written_file(tmp_path: Path) -> None:
    """The answer says ```RUN-RECORD.md``` in backticks beside the folder: flagged, kept."""
    _write(tmp_path / "pkg" / "RUN-RECORD.md")
    items = [call("bash", command="cd pkg; echo x > RUN-RECORD.md")]
    (only,) = _run(items, "Receipt: `RUN-RECORD.md` in `pkg/`.", tmp_path).candidates
    assert only.linked_in_answer is True and only.path == "pkg/RUN-RECORD.md"

"""``scripts/ci_main_red_body.py``: the body must name the diagnosis, not the key.

WHY THESE CASES. The `ci-main-red` issue is what tells the team main is red, and
the first version of that body said only "Full suite red at <sha>. Failing jobs:
- `test`". Answering "infra, flake or regression?" then cost an artifact download
and several tool calls. Measured on run 42463603282 (issue #3617): the cause was
`curl: (22) ... 504` in the `Install ripgrep` step of shard `test (3.12, 1)`,
which never reached pytest -- the body named no step, no error and no node id.

So the three shapes that matter are pinned here, each against a log in the real
`gh run view --job <id> --log` format (`<job>\\t<step>\\t<timestamp> <message>`):

* a job that died in a setup step BEFORE the tests ran (the infra case),
* a shard that ran the suite and has `FAILED tests/...` node ids (the flake or
  regression case),
* a job whose log never arrives (the API said no), which must still be filed.

And because this runs inside the only step that reports main's health, the
best-effort contract is pinned too: a builder that raises still files the thin
body and exits 0.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ci_main_red_body.py"
_spec = importlib.util.spec_from_file_location("ci_main_red_body", SCRIPT)
assert _spec and _spec.loader
body_builder = importlib.util.module_from_spec(_spec)
sys.modules["ci_main_red_body"] = body_builder
_spec.loader.exec_module(body_builder)

SHA = "5d426ecc36a96a028975aa267356cc533f6061f7"
RUN_URL = "https://github.com/damianvtran/local-operator/actions/runs/42463603282"
JOB_URL = RUN_URL + "/job/128684165358"


def _log(job: str, rows: list[tuple[str, str]]) -> str:
    """A captured job log: every line is `<job>\\t<step>\\t<timestamp> <message>`."""
    lines = []
    for index, (step, message) in enumerate(rows):
        lines.append(f"{job}\t{step}\t2026-11-08T09:04:{index:02d}.1234567Z {message}")
    return "\n".join(lines) + "\n"


def _steps(*spec: tuple[str, str]) -> list[dict]:
    """Job metadata steps, in order: (name, conclusion)."""
    return [
        {"number": index + 1, "name": name, "conclusion": conclusion}
        for index, (name, conclusion) in enumerate(spec)
    ]


#: The shapes both fixture logs borrow from the real run, unmodified in form.
_INSTALL_STEPS = _steps(
    ("Set up job", "success"),
    ("Install dependencies", "success"),
    ("Install ripgrep", "failure"),
    ("Partition test suite", "skipped"),
    ("Run test shard with coverage", "skipped"),
)
_SHARD_STEPS = _steps(
    ("Set up job", "success"),
    ("Install dependencies", "success"),
    ("Install ripgrep", "success"),
    ("Partition test suite", "success"),
    ("Run test shard with coverage", "failure"),
)


def _install_log() -> str:
    """#3617: four identical 504s in `Install ripgrep`, then the runner's exit line."""
    curl = "curl: (22) The requested URL returned error: 504"
    return _log(
        "test (3.12, 1)",
        [
            ("Install dependencies", "Successfully installed local-operator"),
            ("Install ripgrep", "##[group]Run bash scripts/ci_install_ripgrep.sh"),
            ("Install ripgrep", "  RG_VERSION: 15.1.0"),
            ("Install ripgrep", "  RG_SHA256: 1c9297be4a084eea7ecaedf93eb03d058d6faae"),
            ("Install ripgrep", "##[endgroup]"),
            ("Install ripgrep", curl),
            ("Install ripgrep", curl),
            ("Install ripgrep", curl),
            ("Install ripgrep", curl),
            ("Install ripgrep", "##[error]Process completed with exit code 22."),
        ],
    )


def _shard_log() -> str:
    """A shard that ran the suite: one node id in pytest's short summary."""
    node = (
        "tests/unit/session/runtime/test_live_inbox_drain.py"
        "::test_a_live_queued_steer_keeps_its_backup_across_later_ticks"
    )
    return _log(
        "test (3.12, 3)",
        [
            ("Run test shard with coverage", "##[group]Run python -m pytest"),
            ("Run test shard with coverage", "##[endgroup]"),
            ("Run test shard with coverage", "=========================== short test summary"),
            (
                "Run test shard with coverage",
                f"FAILED {node} - OSError: inbox snapshot prefix changed",
            ),
            (
                "Run test shard with coverage",
                "=== 1 failed, 10877 passed in 1615.15s (0:26:55) ====",
            ),
            ("Run test shard with coverage", "##[error]Process completed with exit code 1."),
        ],
    )


def _render(failing=("test",), cancelled=(), diagnoses=()) -> str:
    return body_builder.render_body(
        sha=SHA,
        run_url=RUN_URL,
        failing=list(failing),
        cancelled=list(cancelled),
        diagnoses=list(diagnoses),
    )


def test_a_setup_step_failure_is_named_and_called_out_before_the_tests() -> None:
    """#3617's shape: the body has to say "this never reached pytest" by itself."""
    diagnosis = body_builder.diagnose("test (3.12, 1)", JOB_URL, _INSTALL_STEPS, _install_log())
    body = _render(diagnoses=[diagnosis])

    assert "failed in step `Install ripgrep`, before the tests ran" in body
    assert "`Run test shard with coverage` never executed" in body
    assert "setup/infrastructure failure, not a test red" in body
    # The line a triager acts on, and the runner's exit line, both present.
    assert "`curl: (22) The requested URL returned error: 504`" in body
    assert "`Process completed with exit code 22.`" in body
    # The environment echo the runner prints before `##[endgroup]` is not output.
    assert "RG_SHA256" not in body
    assert "failing test node ids" not in body


def test_a_shard_that_ran_the_suite_lists_its_failed_node_ids() -> None:
    """The highest-value field: the node id and its reason, without a download."""
    diagnosis = body_builder.diagnose("test (3.12, 3)", JOB_URL, _SHARD_STEPS, _shard_log())
    body = _render(diagnoses=[diagnosis])

    assert "failed in step `Run test shard with coverage`" in body
    assert "before the tests ran" not in body
    assert "failing test node ids (1)" in body
    assert (
        "`tests/unit/session/runtime/test_live_inbox_drain.py"
        "::test_a_live_queued_steer_keeps_its_backup_across_later_ticks"
        " - OSError: inbox snapshot prefix changed`"
    ) in body
    # The summary count survives as tail output, and the node line is not repeated.
    assert "1 failed, 10877 passed" in body
    assert body.count("test_a_live_queued_steer_keeps_its_backup_across_later_ticks") == 1


def test_identical_repeated_errors_collapse_to_one_line() -> None:
    """Four 504s in one step are one fact, and the tail has five slots."""
    diagnosis = body_builder.diagnose("test (3.12, 1)", JOB_URL, _INSTALL_STEPS, _install_log())
    assert diagnosis.tail == ["curl: (22) The requested URL returned error: 504"]


def test_an_unavailable_log_still_names_the_job_and_links_it() -> None:
    """The API can refuse a log; that must not cost the job its identity."""
    diagnosis = body_builder.diagnose("test (3.12, 1)", JOB_URL, _INSTALL_STEPS, None)
    body = _render(diagnoses=[diagnosis])

    assert "`test (3.12, 1)`" in body
    assert f"[log]({JOB_URL})" in body
    assert "log unavailable for this job" in body
    assert "failed in step `Install ripgrep`" in body


def test_a_log_without_step_metadata_still_yields_nodes() -> None:
    """Without `steps` the failing-step name is unknown; nodes are not."""
    diagnosis = body_builder.diagnose("test (3.12, 3)", JOB_URL, [], _shard_log())
    body = _render(diagnoses=[diagnosis])

    assert diagnosis.failed_step == ""
    assert "failing test node ids (1)" in body
    assert "test_live_inbox_drain" in body


def test_a_failing_test_step_with_no_node_id_says_so() -> None:
    """ "No node id" is a collection error or a crash; it is not "no failure"."""
    rows = [
        ("Run test shard with coverage", "##[endgroup]"),
        ("Run test shard with coverage", "ERROR tests/unit/test_x.py - ImportError"),
        ("Run test shard with coverage", "##[error]Process completed with exit code 1."),
    ]
    diagnosis = body_builder.diagnose(
        "test (3.12, 0)", JOB_URL, _SHARD_STEPS, _log("test (3.12, 0)", rows)
    )
    body = _render(diagnoses=[diagnosis])

    assert "the test step failed without a `FAILED tests/...` node id" in body


def test_node_ids_are_capped_and_the_omitted_count_is_named() -> None:
    """A mass failure must not paste a shard's whole summary into the issue."""
    node = "tests/unit/test_bulk.py::test_case_{}"
    rows = [("Run test shard with coverage", "##[endgroup]")]
    rows += [
        ("Run test shard with coverage", f"FAILED {node.format(i)} - AssertionError: nope")
        for i in range(30)
    ]
    diagnosis = body_builder.diagnose(
        "test (3.12, 0)", JOB_URL, _SHARD_STEPS, _log("test (3.12, 0)", rows)
    )
    body = _render(diagnoses=[diagnosis])

    assert len(diagnosis.nodes) == body_builder.MAX_NODES_PER_JOB == 20
    assert diagnosis.nodes_omitted == 10
    assert "failing test node ids (30, first 20 shown)" in body
    assert "… 10 more, see the log" in body


def test_the_body_is_capped_and_says_where_the_rest_is() -> None:
    """Six jobs x 20 nodes x five tail lines overflows an issue; the cut is stated."""
    diagnoses = []
    for index in range(6):
        rows = [("Run test shard with coverage", "##[endgroup]")]
        rows += [
            (
                "Run test shard with coverage",
                f"FAILED tests/unit/test_bulk.py::test_case_{index}_{n} - " + "x" * 120,
            )
            for n in range(20)
        ]
        rows += [("Run test shard with coverage", f"tail line {n} " + "y" * 200) for n in range(10)]
        diagnoses.append(
            body_builder.diagnose(
                f"test (3.12, {index})",
                JOB_URL,
                _SHARD_STEPS,
                _log(f"test (3.12, {index})", rows),
            )
        )
    body = _render(diagnoses=diagnoses)

    assert len(body) <= body_builder.MAX_BODY_CHARS + 200
    assert "body truncated at 12000 characters" in body
    assert RUN_URL in body


def test_uncapped_body_keeps_the_run_url_last() -> None:
    """The link the whole body hangs off must survive rendering, in every case."""
    diagnosis = body_builder.diagnose("test (3.12, 1)", JOB_URL, _INSTALL_STEPS, _install_log())
    assert _render(diagnoses=[diagnosis]).rstrip().endswith(RUN_URL)


def test_cancelled_jobs_are_still_named_beside_failures() -> None:
    """A shard at its cap produced no verdict; the issue has to say so."""
    body = _render(failing=["test"], cancelled=["tui-e2e"])
    assert "Cancelled jobs:" in body
    assert "- `tui-e2e`" in body


def test_a_failing_key_with_no_job_metadata_is_still_listed() -> None:
    """Metadata fetch can fail; the old body's one useful field must survive."""
    body = _render(failing=["coverage-report", "test"], diagnoses=[])
    assert "- `coverage-report` — no per-job detail available for this job" in body
    assert "- `test` — no per-job detail available for this job" in body


def test_render_from_evidence_picks_the_failing_shard_from_a_captured_payload(
    tmp_path: Path,
) -> None:
    """The offline path is the one the PR's evidence used, so it is pinned."""
    (tmp_path / "jobs.json").write_text(
        json.dumps(
            {
                "total_count": 3,
                "jobs": [
                    {
                        "id": 1,
                        "name": "test (3.12, 0)",
                        "conclusion": "success",
                        "html_url": RUN_URL + "/job/1",
                    },
                    {
                        "id": 2,
                        "name": "test (3.12, 1)",
                        "conclusion": "failure",
                        "html_url": JOB_URL,
                        "steps": _INSTALL_STEPS,
                    },
                    {
                        "id": 3,
                        "name": "coverage-report",
                        "conclusion": "success",
                        "html_url": RUN_URL + "/job/3",
                    },
                ],
            }
        )
    )
    (tmp_path / "2.log").write_text(_install_log())

    body = body_builder.render_from_evidence(
        sha=SHA, run_url=RUN_URL, failing=["test"], cancelled=[], evidence=tmp_path
    )

    assert "`test (3.12, 1)`" in body
    assert "`test (3.12, 0)`" not in body
    assert "Install ripgrep" in body


def test_a_builder_that_raises_still_files_the_thin_body(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The whole point: enrichment is optional, filing is not."""
    out = tmp_path / "body.md"

    def _boom(*args, **kwargs):
        raise RuntimeError("gh exploded")

    monkeypatch.setattr(body_builder, "collect", _boom)
    rc = body_builder.main(
        [
            "--sha",
            SHA,
            "--run-url",
            RUN_URL,
            "--needs-json",
            json.dumps({"test": {"result": "failure"}}),
            "--repo",
            "damianvtran/local-operator",
            "--run-id",
            "42463603282",
            "--out",
            str(out),
        ]
    )

    assert rc == 0, "a body builder failure must not fail the filing step"
    body = out.read_text()
    assert body == (f"Full suite red at {SHA}.\n\nFailing jobs:\n- `test`\n\nRun: {RUN_URL}\n")
    assert "ci-main-red body enrichment failed" in capsys.readouterr().err


def test_an_empty_needs_payload_never_claims_that_nothing_failed(tmp_path: Path, capsys) -> None:
    """A body reading "Failing jobs:" with none listed is worse than the thin one."""
    out = tmp_path / "body.md"
    rc = body_builder.main(
        [
            "--sha",
            SHA,
            "--run-url",
            RUN_URL,
            "--needs-json",
            "{}",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    body = out.read_text()
    assert "- (none reported)" in body
    assert "enrichment skipped" in capsys.readouterr().err


def test_only_six_logs_are_fetched_and_the_rest_say_so(tmp_path: Path) -> None:
    """Bounded fetching is fine; a silently missing job is not."""
    jobs = [
        {
            "id": index,
            "name": f"test (3.12, {index})",
            "conclusion": "failure",
            "html_url": f"{RUN_URL}/job/{index}",
            "steps": _SHARD_STEPS,
        }
        for index in range(7)
    ]
    (tmp_path / "jobs.json").write_text(json.dumps({"jobs": jobs}))
    for index in range(7):
        (tmp_path / f"{index}.log").write_text(_shard_log())

    body = body_builder.render_from_evidence(
        sha=SHA, run_url=RUN_URL, failing=["test"], cancelled=[], evidence=tmp_path
    )

    assert body.count("no log fetched") == 1, "only the seventh job is beyond the cap"
    assert "`test (3.12, 6)`" in body, "a failing job must never vanish from the body"


def test_the_report_job_runs_the_builder_and_keeps_a_thin_body_fallback() -> None:
    """The workflow wiring is the contract; deleting it silently reverts the body."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "full-suite.yml").read_text()

    assert "python3 scripts/ci_main_red_body.py" in workflow
    # The fallback is what keeps a broken builder from leaving main red and
    # unnoticed, so both halves are required.
    assert "Full suite red at %s." in workflow
    assert "if ! python3 scripts/ci_main_red_body.py" in workflow
    # Reading a job's log is an Actions API call; without this scope the
    # enrichment 403s and every body silently degrades.
    assert "actions: read" in workflow

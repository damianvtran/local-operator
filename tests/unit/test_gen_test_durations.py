"""Fail-closed timing ingestion: missing attribution must not hide heavy tests."""

import base64
import io
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import gen_test_durations as generator


def test_nested_classes_and_parameterized_names_resolve_new_files(tmp_path: Path) -> None:
    # Resolve against the source tree, never the stale manifest we are replacing.
    module = tmp_path / "tests/unit/test_new.py"
    module.parent.mkdir(parents=True)
    module.touch()
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.unit.test_new.TestOuter.TestInner" '
        'name="test_case[value.with.dots]" time="7.5"/></testsuite>'
    )
    assert generator.aggregate([report], repo=tmp_path) == {"tests/unit/test_new.py": 7.5}


def test_unattributed_duration_is_fatal(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.unit.test_missing" time="123"/></testsuite>'
    )
    with pytest.raises(ValueError, match="test_missing"):
        generator.aggregate([report], repo=tmp_path)


@pytest.mark.parametrize("conclusion", ["cancelled", "failure", "timed_out"])
def test_incomplete_run_cannot_supply_weights(
    conclusion: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            {"status": "completed", "conclusion": conclusion},
            {"jobs": []},
            {"total_count": 0, "artifacts": []},
        ]
    )
    monkeypatch.setattr(generator, "_gh_json", lambda _: next(responses))
    download = Mock()
    monkeypatch.setattr(generator.subprocess, "check_output", download)
    with pytest.raises(ValueError, match="did not complete successfully"):
        generator.ci_run(123, 1, tmp_path)
    download.assert_not_called()


@pytest.mark.parametrize("expected,observed", [(5, 5), (6, 6), (6, 5)])
def test_attempt_interval_excludes_other_attempt_artifacts(
    expected: int, observed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generator, "_source_matrix", lambda *_: (list(range(expected)), {}))
    jobs, artifacts = [], []
    for shard in range(observed):
        jobs.append(
            {
                "id": 100 + shard,
                "name": f"test (3.12, {shard})",
                "conclusion": "success",
                "started_at": "2026-11-01T01:00:00Z",
                "completed_at": "2026-11-01T02:00:00Z",
            }
        )
        for attempt, created in [(1, "2026-11-01T00:30:00Z"), (2, "2026-11-01T01:30:00Z")]:
            artifacts.append(
                {
                    "id": attempt * 10 + shard,
                    "name": f"junit-timings-3.12-{shard}",
                    "created_at": created,
                    "expired": False,
                }
            )
    responses = iter(
        [
            {
                "status": "completed",
                "conclusion": "success",
                "head_sha": "abc",
                "path": ".github/workflows/ci.yml",
            },
            {"jobs": jobs},
            {"total_count": len(artifacts), "artifacts": artifacts},
        ]
    )
    monkeypatch.setattr(generator, "_gh_json", lambda _: next(responses))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for filename in [f"junit-3.12-{s}.xml" for s in range(observed)] + [
            "junit-serial-controls-3.12-0.xml"
        ]:
            zipped.writestr(
                filename,
                '<testsuite><testcase classname="tests.unit.test_paths" '
                'name="test_x[param.with.dots]" time="2"/></testsuite>',
            )
    download = Mock(return_value=archive.getvalue())
    monkeypatch.setattr(generator.subprocess, "check_output", download)
    if expected != observed:
        with pytest.raises(ValueError, match="incomplete or unsuccessful shard matrix"):
            generator.ci_run(123, 2, tmp_path)
        download.assert_not_called()
        return
    totals, provenance = generator.ci_run(123, 2, tmp_path)
    assert totals == {"tests/unit/test_paths.py": 2.0 * (observed + 1)}  # includes serial
    assert provenance["attempt"] == 2
    assert [source["artifact_id"] for source in provenance["shards"]] == list(
        range(20, 20 + observed)
    )
    assert all(
        str(20 + s) + "/zip" in call.args[0][-1] for s, call in enumerate(download.call_args_list)
    )


@pytest.mark.parametrize(
    "identities",
    [
        ["1:1", "2:1"],
        ["1:1", "1:2", "2:1"],
        ["bad"],
        ["1:0"],
    ],
)
def test_ci_mode_rejects_unqualified_inputs_before_fetching(
    identities: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch = Mock()
    monkeypatch.setattr(generator, "ci_run", fetch)
    args = ["--cache-dir", str(tmp_path)]
    for identity in identities:
        args.extend(["--ci-run", identity])
    with pytest.raises(SystemExit) as error:
        generator.main(args)
    assert error.value.code == 2
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "matrix",
    [
        "shard: ${{ fromJSON(needs.source.outputs.shards) }}",
        "shard: [0, 1, 2, 3, 4]\n        shard: [0, 1, 2, 3, 4, 5]",
        "shard: [0, 1, 2, 3, 4]\n        exclude: [{shard: 4}]",
        "shard: [0, 1, 2, 3, 4]\n        include: [{shard: 5}]",
        "shard: [0, 1, 1, 2, 3]",
        'shard: ["0", 1, 2, 3, 4]',
        "shard: [0, 1, 2, 3]",
        "shard: [",
        "shard: &aliased [0, 1, 2, 3, 4]",
    ],
)
def test_ambiguous_or_dynamic_source_matrix_fails_closed(matrix: str) -> None:
    workflow = "jobs:\n  test:\n    strategy:\n      matrix:\n        " + matrix + "\n"
    with pytest.raises(ValueError, match="source workflow"):
        generator._workflow_shards(workflow)


@pytest.mark.parametrize("count", [5, 6])
def test_literal_source_matrix_is_read_without_evaluating_expressions(count: int) -> None:
    workflow = (
        "jobs:\n  test:\n    strategy:\n      matrix:\n"
        "        python-version: ${{ arbitrary.expression }}\n"
        f"        shard: {list(range(count))}\n"
    )
    assert generator._workflow_shards(workflow) == list(range(count))


MERGE_SHA = "a" * 40
HEAD_SHA = "b" * 40
CHECKOUT_LOG = (
    "##[group]Checking out the ref\n"
    "[command]/usr/bin/git checkout --force refs/remotes/pull/1/merge\n"
    "##[endgroup]\n[command]/usr/bin/git log -1 --format=%H\n" + MERGE_SHA + "\n"
)


def test_source_uses_checkout_merge_not_branch_or_unrelated_log_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = "c" * 40 + "\n" + CHECKOUT_LOG + "[command]/usr/bin/git log -1 --format=%H\n" + "d" * 40
    monkeypatch.setattr(generator.subprocess, "check_output", Mock(return_value=log))
    workflow = "jobs:\n  test:\n    strategy:\n      matrix:\n        shard: [0, 1, 2, 3, 4, 5]\n"
    fetch = Mock(
        side_effect=[
            {"parents": [{"sha": HEAD_SHA}, {"sha": "e" * 40}]},
            {
                "sha": "blob",
                "encoding": "base64",
                "content": base64.b64encode(workflow.encode()).decode(),
            },
        ]
    )
    monkeypatch.setattr(generator, "_gh_json", fetch)
    shards, source = generator._source_matrix(
        {"event": "pull_request", "head_sha": HEAD_SHA}, {"id": 42}
    )
    assert shards == list(range(6))
    assert source == {"workflow_sha": MERGE_SHA, "workflow_blob": "blob"}
    assert fetch.call_args_list[0].args[0].endswith("/" + MERGE_SHA)
    assert fetch.call_args_list[1].args[0].endswith("?ref=" + MERGE_SHA)


@pytest.mark.parametrize(
    "log",
    [
        "c" * 40,
        CHECKOUT_LOG + CHECKOUT_LOG,
        "[command]/usr/bin/git log -1 --format=%H\n" + MERGE_SHA,
        "##[group]Checking out the ref\n##[endgroup]\nmissing identity",
    ],
)
def test_missing_or_ambiguous_checkout_identity_rejects(
    log: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generator.subprocess, "check_output", Mock(return_value=log))
    fetch = Mock()
    monkeypatch.setattr(generator, "_gh_json", fetch)
    with pytest.raises(ValueError, match="checkout SHA"):
        generator._source_matrix({"event": "pull_request", "head_sha": HEAD_SHA}, {"id": 42})
    fetch.assert_not_called()


def test_wrong_merge_ancestry_rejects_before_source_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator.subprocess, "check_output", Mock(return_value=CHECKOUT_LOG))
    fetch = Mock(return_value={"parents": [{"sha": "c" * 40}, {"sha": "d" * 40}]})
    monkeypatch.setattr(generator, "_gh_json", fetch)
    with pytest.raises(ValueError, match="does not contain run head"):
        generator._source_matrix({"event": "pull_request", "head_sha": HEAD_SHA}, {"id": 42})
    assert fetch.call_count == 1


@pytest.mark.parametrize("failure_at", ["log", "commit", "source"])
def test_source_api_failures_never_fall_back(
    failure_at: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = generator.subprocess.CalledProcessError(1, "gh")
    monkeypatch.setattr(
        generator.subprocess,
        "check_output",
        Mock(side_effect=failure if failure_at == "log" else None, return_value=CHECKOUT_LOG),
    )
    fetch = Mock(
        side_effect=(
            [failure]
            if failure_at == "commit"
            else [{"parents": [{"sha": HEAD_SHA}, {"sha": "e" * 40}]}, failure]
        )
    )
    monkeypatch.setattr(generator, "_gh_json", fetch)
    with pytest.raises(generator.subprocess.CalledProcessError):
        generator._source_matrix({"event": "pull_request", "head_sha": HEAD_SHA}, {"id": 42})


@pytest.mark.parametrize("declared,observed", [(5, 5), (6, 6), (6, 5)])
def test_source_matrix_completeness_through_cli(
    declared: int, observed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole CLI must reject a missing sixth shard before writing p75."""
    workflow = (
        "jobs:\n  test:\n    strategy:\n      matrix:\n" f"        shard: {list(range(declared))}\n"
    )

    def metadata(endpoint: str) -> dict[str, object]:
        if "/git/commits/" in endpoint:
            return {"parents": [{"sha": HEAD_SHA}, {"sha": "e" * 40}]}
        if "/contents/" in endpoint:
            return {
                "sha": "blob",
                "encoding": "base64",
                "content": base64.b64encode(workflow.encode()).decode(),
            }
        if "/jobs?" in endpoint:
            return {
                "jobs": [
                    {
                        "id": 100 + s,
                        "name": f"test (3.12, {s})",
                        "conclusion": "success",
                        "started_at": "2026-11-01T01:00:00Z",
                        "completed_at": "2026-11-01T02:00:00Z",
                    }
                    for s in range(observed)
                ]
            }
        if "/artifacts?" in endpoint:
            return {
                "total_count": observed,
                "artifacts": [
                    {
                        "id": 20 + s,
                        "name": f"junit-timings-3.12-{s}-attempt-1",
                        "created_at": "2026-11-01T01:30:00Z",
                        "expired": False,
                    }
                    for s in range(observed)
                ],
            }
        return {
            "status": "completed",
            "conclusion": "success",
            "event": "pull_request",
            "head_sha": HEAD_SHA,
            "path": ".github/workflows/ci.yml",
        }

    monkeypatch.setattr(generator, "_gh_json", metadata)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for filename in [f"junit-3.12-{s}.xml" for s in range(observed)] + [
            "junit-serial-controls-3.12-0.xml"
        ]:
            zipped.writestr(
                filename,
                '<testsuite><testcase classname="tests.unit.test_paths" time="2"/></testsuite>',
            )
    download = Mock(
        side_effect=lambda args, **kwargs: (
            CHECKOUT_LOG if args[-1].endswith("/logs") else archive.getvalue()
        )
    )
    monkeypatch.setattr(generator.subprocess, "check_output", download)
    out = tmp_path / "durations.json"
    args = ["--cache-dir", str(tmp_path / "cache"), "--out", str(out)]
    for run_id in (1, 2, 3):
        args.extend(["--ci-run", f"{run_id}:1"])
    if declared != observed:
        with pytest.raises(ValueError, match="expects \\[0, 1, 2, 3, 4, 5\\]"):
            generator.main(args)
        assert not out.exists()
        assert all(call.args[0][-1].endswith("/logs") for call in download.call_args_list)
    else:
        assert generator.main(args) == 0
        manifest = generator.json.loads(out.read_text())
        assert manifest["durations"] == {"tests/unit/test_paths.py": 2.0 * (declared + 1)}
        assert all(len(run["shards"]) == declared for run in manifest["_provenance"]["runs"])


def test_p75_preserves_files_absent_from_other_runs() -> None:
    assert generator.percentile_weights([{"tests/unit/test_new.py": 8.0}, {}, {}]) == {
        "tests/unit/test_new.py": 8.0
    }


@pytest.mark.parametrize("duration", ["nan", "inf", "-1"])
def test_invalid_durations_are_fatal(duration: str, tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.unit.test_paths" ' f'time="{duration}"/></testsuite>'
    )
    with pytest.raises(ValueError, match="invalid duration"):
        generator.aggregate([report])


def test_p75_uses_per_run_file_totals() -> None:
    # Summing testcases across runs or averaging them hides a costly tail.
    assert generator.percentile_weights(
        [
            {"tests/unit/test_new.py": 1.0},
            {"tests/unit/test_new.py": 5.0},
            {"tests/unit/test_new.py": 9.0},
        ]
    ) == {"tests/unit/test_new.py": 7.0}

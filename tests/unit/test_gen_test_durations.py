"""Fail-closed timing ingestion: missing attribution must not hide heavy tests."""

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


def test_attempt_interval_excludes_other_attempt_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, artifacts = [], []
    for shard in range(5):
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
            {"total_count": 10, "artifacts": artifacts},
        ]
    )
    monkeypatch.setattr(generator, "_gh_json", lambda _: next(responses))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for filename in [f"junit-3.12-{s}.xml" for s in range(5)] + [
            "junit-serial-controls-3.12-0.xml"
        ]:
            zipped.writestr(
                filename,
                '<testsuite><testcase classname="tests.unit.test_paths" '
                'name="test_x[param.with.dots]" time="2"/></testsuite>',
            )
    download = Mock(return_value=archive.getvalue())
    monkeypatch.setattr(generator.subprocess, "check_output", download)
    totals, provenance = generator.ci_run(123, 2, tmp_path)
    assert totals == {"tests/unit/test_paths.py": 12.0}  # serial XML is included
    assert provenance["attempt"] == 2
    assert [source["artifact_id"] for source in provenance["shards"]] == list(range(20, 25))
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

"""The local run record: written once, updated in place, and free of secrets.

WHAT IS PROVEN HERE. That a run keeps a row across a real process boundary (a
subprocess reads the file a test's process wrote), that a status read updates
that row instead of appending a second one, that a deleted run stays visible as
deleted, that the file survives a torn or hand-edited line, and — the one that
matters most — that it holds no credential: the test scans the bytes for the
bearer this device sent and for every credential SHAPE the repo's scrubber knows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.redaction_shapes import match_shape_names, scrub_shapes
from local_operator.remote_compute import records

#: A synthetic bearer with no meaning anywhere. It must never appear in the
#: record — it is what a record written straight from a request would leak.
BEARER = "zq-bearer-7c1f9a"
#: A prompt that quotes credentials, as an operator's can.
PROMPTY = (
    "please fix the deploy\n"
    f"it uses AWS_SECRET_ACCESS_KEY={BEARER}\n"
    "dsn mongodb+srv://svc:zq-pass-91@cluster.example.invalid/db\n"
    "\t trailing   whitespace   "
)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    root = tmp_path / "config"
    root.mkdir()
    return root


def _record(
    root: Path,
    *,
    run_id: str = "run_1",
    prompt: str = "fix the flaky test",
    network_id: str | None = "net_1",
    status: str | None = "pending",
    api_origin: str = "https://api.example.invalid",
) -> records.RunRecord:
    return records.record_submission(
        run_id=run_id,
        repo_url="https://example.invalid/repo.git",
        repo_sha="0" * 40,
        prompt=prompt,
        size="m",
        api_origin=api_origin,
        network_id=network_id,
        status=status,
        config_dir=root,
    )


def test_a_submission_is_written_with_what_was_submitted(root: Path) -> None:
    stored = _record(root)
    assert stored.created_at.endswith("Z")
    assert stored.prompt_summary == "fix the flaky test"
    assert stored.api_origin == "https://api.example.invalid"
    assert records.load(root) == [stored]


def test_the_record_lives_under_the_config_directory(root: Path) -> None:
    path = records.records_path(root)
    assert path == root / "remote-compute" / "runs.jsonl"
    _record(root)
    assert path.is_file()


def test_a_status_read_updates_the_row_instead_of_appending_a_second_run(root: Path) -> None:
    _record(root)
    updated = records.record_status(
        {"id": "run_1", "status": "running"},
        api_origin="https://api.example.invalid",
        config_dir=root,
    )
    assert updated is not None
    assert updated.status == "running"
    assert updated.updated_at is not None
    listed = records.load(root)
    assert [row.run_id for row in listed] == ["run_1"]
    assert listed[0].status == "running"


def test_a_status_read_of_an_unrecorded_run_writes_nothing(root: Path) -> None:
    _record(root)
    assert (
        records.record_status(
            {"id": "run_other", "status": "running"},
            api_origin="https://api.example.invalid",
            config_dir=root,
        )
        is None
    )
    assert [row.run_id for row in records.load(root)] == ["run_1"]


def test_a_deleted_run_stays_in_the_list_as_deleted(root: Path) -> None:
    _record(root)
    deleted = records.mark_deleted("run_1", config_dir=root)
    assert deleted is not None
    assert deleted.deleted is True
    assert records.load(root)[0].deleted is True


def test_the_record_survives_a_restart(root: Path) -> None:
    """A FRESH PROCESS reads what this one wrote — the panel's offline answer.

    A test that only re-read the file in its own process would prove the fold,
    not the format: this spawns the interpreter the gates run with and asks it
    to load the record from the same config directory.
    """
    _record(root, run_id="run_restart", status="running")
    environment = dict(os.environ)
    environment["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
    program = (
        "import json;"
        "from local_operator.remote_compute import records;"
        "print(json.dumps([[r.run_id, r.status] for r in records.load()]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [["run_restart", "running"]]


def test_the_file_holds_no_bearer_and_no_credential_shape(root: Path) -> None:
    """The prompt is free text, so the summary is scrubbed before it is stored."""
    _record(root, prompt=PROMPTY)
    raw = records.records_path(root).read_text(encoding="utf-8")
    assert BEARER not in raw
    assert "zq-pass-91" not in raw
    assert match_shape_names(raw) == []
    assert scrub_shapes(raw) == raw


def test_the_summary_is_one_line_and_capped() -> None:
    summary = records.prompt_summary("a" * 500 + "\n\nb\tc")
    assert len(summary) == records.SUMMARY_CHARS
    assert "\n" not in summary
    assert "\t" not in summary


def test_a_torn_or_hand_edited_line_is_skipped(root: Path) -> None:
    _record(root, run_id="run_1")
    path = records.records_path(root)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"run_id": "run_2", "repo_url"')  # a crash mid-append
        handle.write("\n")
        handle.write('{"nonsense": true}\n')
    assert [row.run_id for row in records.load(root)] == ["run_1"]


def test_a_line_from_an_older_build_is_ignored_rather_than_guessed(root: Path) -> None:
    path = records.records_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"run_id": "run_old", "repo_url": "https://x.invalid/r.git"}) + "\n",
        encoding="utf-8",
    )
    assert records.load(root) == []


def test_the_newest_run_is_listed_first(root: Path) -> None:
    first = _record(root, run_id="run_1")
    second = _record(root, run_id="run_2")
    assert first.created_at <= second.created_at
    assert [row.run_id for row in records.load(root)] == ["run_2", "run_1"]


def test_the_file_is_compacted_once_it_passes_its_size_bound(root: Path) -> None:
    """Compaction is the file's only bound; without it appends grow forever."""
    total = records.KEPT_RUNS + 10
    for index in range(total):
        _record(root, run_id=f"run_{index:04d}")
    path = records.records_path(root)
    # Enough observations to cross the byte bound: each line is ~650 bytes, so
    # three status reads per run takes the file past 1 MB.
    for _ in range(3):
        for index in range(total):
            records.record_status(
                {"id": f"run_{index:04d}", "status": "running"},
                api_origin="https://api.example.invalid",
                config_dir=root,
            )
    # The bound is BYTES: one line may land after the rewrite that crossed it.
    assert path.stat().st_size <= records.MAX_FILE_BYTES + 1024
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # ...and a rewrite did happen: far fewer lines than the appends made.
    assert len(lines) < total * 4
    listed = records.load(root)
    assert records.KEPT_RUNS <= len(listed) <= records.KEPT_RUNS + 1
    assert all(row.status == "running" for row in listed)


def test_the_newest_runs_survive_a_compaction(root: Path) -> None:
    """A rewrite keeps the newest rows, not an arbitrary KEPT_RUNS slice."""
    total = records.KEPT_RUNS + 10
    for index in range(total):
        _record(root, run_id=f"run_{index:04d}")
    path = records.records_path(root)
    path.write_text(
        path.read_text(encoding="utf-8") + ("{" + '"pad": "' + "x" * 900 + '"}\n') * 1200,
        encoding="utf-8",
    )
    _record(root, run_id="run_newest")
    listed = records.load(root)
    assert listed[0].run_id == "run_newest"
    assert records.KEPT_RUNS <= len(listed) <= records.KEPT_RUNS + 1


def test_a_missing_file_is_an_empty_list_not_a_failure(root: Path) -> None:
    assert records.load(root) == []

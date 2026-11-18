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
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.redaction_shapes import match_shape_names, scrub_shapes
from local_operator.remote_compute import records

#: A synthetic bearer with no meaning anywhere. It must never appear in the
#: record — it is what a record written straight from a request would leak.
BEARER = "zq-bearer-7c1f9a"
#: How long the compaction test waits for the child process to reach its append.
#: The child boots an interpreter and imports the package (measured here at
#: ~0.4-0.8 s), so this is that with headroom; under the lock the wait runs to
#: the bound, which is the price of making the window deterministic.
WAIT_FOR_CHILD_S = 3.0

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


# ---------------------------------------------------------------------------
# Owner-only mode, one line boundary, and the lock the fold shares with an append
# ---------------------------------------------------------------------------


def test_the_record_and_its_directory_are_owner_only(root: Path) -> None:
    """A stored line can quote a prompt, so nobody but the owner may read it.

    The default config root is already 0700; this holds for the case that
    motivated the transcript's identical fix — a copy, a backup, or a
    ``LOCAL_OPERATOR_CONFIG_DIR`` on a shared path.
    """
    _record(root)
    path = records.records_path(root)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_a_compaction_leaves_the_record_owner_only(root: Path) -> None:
    """A rewrite must not widen the mode: the temp file is what becomes the file."""
    for index in range(records.KEPT_RUNS + 5):
        _record(root, run_id=f"run_{index:04d}")
    path = records.records_path(root)
    path.write_text(
        path.read_text(encoding="utf-8") + ('{"pad": "' + "x" * 900 + '"}\n') * 1200,
        encoding="utf-8",
    )
    _record(root, run_id="run_newest")
    assert [row.run_id for row in records.load(root)][0] == "run_newest"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_world_readable_record_is_tightened_on_its_next_append(root: Path) -> None:
    """An older build's file, a backup or an import becomes private on next write."""
    path = records.records_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    _record(root)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_torn_final_line_does_not_swallow_the_next_append(root: Path) -> None:
    """One crash must cost one observation, not two.

    A crash mid-write leaves a line with no newline; appending onto it makes the
    NEW line unparsable as well, so the run just submitted is the one that
    disappears. The append therefore starts a line boundary first.
    """
    path = records.records_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"run_id": "run_torn"', encoding="utf-8")
    _record(root, run_id="run_after")
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith('{"run_id": "run_torn"\n')
    assert raw.endswith("\n")
    assert [row.run_id for row in records.load(root)] == ["run_after"]


def test_an_append_during_a_compaction_is_not_lost(root: Path) -> None:
    """The lost-append window is REPRODUCED, not raced for.

    Without a lock, an append landing between the fold and the ``os.replace`` is
    in neither the old file (replaced) nor the new one (folded before the line
    existed), so that run never appears in the panel and nothing reports an
    error. The window is opened deterministically here: ``mkstemp`` spawns a REAL
    child process once the fold has already happened, and ``os.replace`` waits
    until the child's line is readable in the file it is about to replace. Under
    the exclusive lock that line never appears (the child is held at the lock, so
    the wait runs to its bound) and its row is in the file afterwards; with
    ``records._exclusive`` replaced by a no-op — the mutation this test was
    checked against — the same sequence loses the row.
    """
    for index in range(3):
        _record(root, run_id=f"run_{index}")
    marker = root / "child-started"
    environment = dict(os.environ)
    environment["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
    child_program = (
        "import pathlib;"
        "from local_operator.remote_compute import records;"
        f"pathlib.Path({str(marker)!r}).write_text('1');"
        "records.record_submission("
        "run_id='run_child', repo_url='https://example.invalid/r.git',"
        "repo_sha='0' * 40, prompt='from the child', size='m',"
        "api_origin='https://api.example.invalid')"
    )
    started: list[subprocess.Popen[str]] = []
    real_mkstemp = tempfile.mkstemp
    real_replace = os.replace

    def mkstemp_spawning_the_child(**keywords: Any) -> tuple[int, str]:
        handle, temporary = real_mkstemp(**keywords)
        if not started:
            started.append(
                subprocess.Popen(
                    [sys.executable, "-c", child_program],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        return handle, temporary

    def replace_after_the_child_appends(source: Path | str, destination: Path | str) -> None:
        deadline = time.monotonic() + WAIT_FOR_CHILD_S
        while time.monotonic() < deadline:
            try:
                if "run_child" in Path(destination).read_text(encoding="utf-8"):
                    break
            except OSError:
                break
            time.sleep(0.01)
        real_replace(source, destination)

    with pytest.MonkeyPatch.context() as patch:
        # 0 bytes makes every append cross the bound, so the append below is the
        # one that compacts and this test owns the window it opens.
        patch.setattr(records, "MAX_FILE_BYTES", 0)
        patch.setattr(records.tempfile, "mkstemp", mkstemp_spawning_the_child)
        patch.setattr(records.os, "replace", replace_after_the_child_appends)
        _record(root, run_id="run_parent")
    assert started, "the child was never spawned"
    child = started[0]
    _, error = child.communicate(timeout=WAIT_FOR_CHILD_S * 12)
    assert child.returncode == 0, error
    listed = [row.run_id for row in records.load(root)]
    assert "run_child" in listed
    assert "run_parent" in listed


def test_the_stored_repo_url_carries_no_credentials(root: Path) -> None:
    """The scrub is the FILE's contract, not a courtesy of one caller.

    ``repo.detect_repo`` refuses a userinfo origin, but a caller that passes a
    ``--repo``-style URL straight to the record never goes through it.
    """
    stored = records.record_submission(
        run_id="run_secret",
        repo_url="https://user:ghp_supersecret@example.invalid/team/repo.git",
        repo_sha="0" * 40,
        prompt="fix the flaky test",
        size="m",
        api_origin="https://api.example.invalid",
        config_dir=root,
    )
    assert stored.repo_url == "https://example.invalid/team/repo.git"
    raw = records.records_path(root).read_text(encoding="utf-8")
    assert "hunter2-dummy" not in raw
    assert "user" not in json.loads(raw.splitlines()[0])["repo_url"]

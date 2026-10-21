"""The store census: deletions we did not perform become visible.

The measured incident shape this covers: directories removed by something
outside the harness leave no ``.cleanup-log.jsonl`` row and no removal warning.
The census is read-only over the store; it reports a missing directory or a
materially shrunken pad as ``unaccounted-*`` against the last baseline, and a
first run only writes the baseline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from local_operator.session import census
from local_operator.session.cleanup import CLEANUP_LOG_NAME, mark_store, remove_session_dir


def _session(root: Path, name: str, *, transcript: int = 200, pad: int = 0) -> Path:
    directory = root / "sessions" / name
    directory.mkdir(parents=True, exist_ok=True)
    if transcript:
        (directory / "transcript.jsonl").write_bytes(b"x" * transcript)
    if pad:
        scratch = directory / "scratchpad"
        scratch.mkdir()
        (scratch / "build.bin").write_bytes(b"y" * pad)
    return directory


def _rows(root: Path) -> list[dict[str, Any]]:
    path = root / "sessions" / CLEANUP_LOG_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _unaccounted(root: Path, policy: str) -> list[dict[str, Any]]:
    return [row for row in _rows(root) if row.get("policy") == policy]


def test_first_run_writes_a_baseline_and_no_rows(tmp_path: Path) -> None:
    _session(tmp_path, "one")
    _session(tmp_path, "two", pad=64)

    result = census.run_store_census(tmp_path)

    assert result["count"] == 2 and result["reported"] == 0
    assert result["complete"] is True and result["enumerated"] is True
    assert _rows(tmp_path) == [], "a baseline is not a finding"
    payload = json.loads((tmp_path / "sessions" / census.CENSUS_NAME).read_text())
    assert payload["schema"] == census.CENSUS_SCHEMA
    assert set(payload["sessions"]) == {"one", "two"}
    assert payload["sessions"]["two"]["pad_bytes"] == 64
    assert payload["count"] == 2 and payload["enumerated"] is True


def test_an_externally_removed_directory_is_reported_with_its_last_contents(
    tmp_path: Path,
) -> None:
    _session(tmp_path, "kept")
    gone = _session(tmp_path, "gone", transcript=300, pad=128)
    assert census.run_store_census(tmp_path)["count"] == 2
    baseline = json.loads((tmp_path / "sessions" / census.CENSUS_NAME).read_text())
    assert baseline["sessions"]["gone"]["bytes"] >= 428

    shutil.rmtree(gone)  # outside the harness: no receipt, no warning

    result = census.run_store_census(tmp_path)

    assert result["reported"] == 1
    rows = _unaccounted(tmp_path, census.UNACCOUNTED_REMOVAL_POLICY)
    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "gone"
    assert row["reason"] == "session directory vanished without a removal record"
    assert row["actor"] == "census"
    assert row["path"] == str(gone.absolute())
    assert row["live"] is False
    assert row["transcript_bytes"] == 300
    assert row["pad_bytes"] == 128
    assert row["bytes"] >= 428
    assert row["entries"] >= 3

    # Reported once: the next census no longer has the id in its baseline.
    census.run_store_census(tmp_path)
    assert len(_unaccounted(tmp_path, census.UNACCOUNTED_REMOVAL_POLICY)) == 1


def test_a_material_pad_shrink_with_no_row_is_reported(tmp_path: Path) -> None:
    target = _session(tmp_path, "shrinker", pad=2 * 1024 * 1024)
    census.run_store_census(tmp_path)

    # Outside the harness: the audited content pass would have written a
    # delegated_scratchpad row before taking anything.
    (target / "scratchpad" / "build.bin").unlink()

    result = census.run_store_census(tmp_path)

    assert result["reported"] == 1
    rows = _unaccounted(tmp_path, census.UNACCOUNTED_CONTENT_POLICY)
    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "shrinker"
    assert row["before"]["pad_bytes"] == 2 * 1024 * 1024
    assert row["after"]["pad_bytes"] == 0
    assert row["transcript_vanished"] is False
    assert row["live"] is False


def test_a_vanished_transcript_is_reported(tmp_path: Path) -> None:
    target = _session(tmp_path, "lostrows", transcript=512)
    census.run_store_census(tmp_path)
    (target / "transcript.jsonl").unlink()

    census.run_store_census(tmp_path)

    rows = _unaccounted(tmp_path, census.UNACCOUNTED_CONTENT_POLICY)
    assert len(rows) == 1
    assert rows[0]["transcript_vanished"] is True
    assert rows[0]["before"]["transcript_bytes"] == 512
    assert rows[0]["after"]["transcript_bytes"] == 0


def test_a_recorded_removal_produces_no_unaccounted_row(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    mark_store(sessions)
    target = _session(tmp_path, "recorded", pad=32)
    census.run_store_census(tmp_path)
    assert remove_session_dir(
        target,
        config_dir=tmp_path,
        policy="delegated_max_age",
        reason="idle over 48h",
        actor="test",
    )

    census.run_store_census(tmp_path)

    assert _unaccounted(tmp_path, census.UNACCOUNTED_REMOVAL_POLICY) == []
    recorded = [row for row in _rows(tmp_path) if row["policy"] == "delegated_max_age"]
    assert len(recorded) == 1 and recorded[0]["session"] == "recorded"


def test_a_live_sessions_pad_change_is_not_reported(
    tmp_path: Path, monkeypatch: Any
) -> None:
    target = _session(tmp_path, "busy", pad=2 * 1024 * 1024)
    census.run_store_census(tmp_path)
    (target / "scratchpad" / "build.bin").unlink()
    monkeypatch.setattr(census, "_is_live", lambda *_args, **_kwargs: True)

    census.run_store_census(tmp_path)

    assert _unaccounted(tmp_path, census.UNACCOUNTED_CONTENT_POLICY) == []

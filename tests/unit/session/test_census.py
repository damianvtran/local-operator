"""The store census: deletions we did not perform become visible.

The measured incident shape this covers: directories removed by something
outside the harness leave no ``.cleanup-log.jsonl`` row and no removal warning.
The census is read-only over the store; it reports a missing directory or a
materially shrunken pad as ``unaccounted-*`` against the last baseline, and a
first run only writes the baseline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import census
from local_operator.session.cleanup import (
    CLEANUP_LOG_NAME,
    mark_store,
    remove_session_dir,
)


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


def test_a_live_sessions_pad_change_is_not_reported(tmp_path: Path, monkeypatch: Any) -> None:
    target = _session(tmp_path, "busy", pad=2 * 1024 * 1024)
    census.run_store_census(tmp_path)
    (target / "scratchpad" / "build.bin").unlink()
    monkeypatch.setattr(census, "_is_live", lambda *_args, **_kwargs: True)

    census.run_store_census(tmp_path)

    assert _unaccounted(tmp_path, census.UNACCOUNTED_CONTENT_POLICY) == []


def test_an_unreadable_pad_is_not_reported_as_a_loss(tmp_path: Path, caplog: object) -> None:
    """Review round 1, M1: a 0000 subtree must not become evidence of deletion."""
    target = _session(tmp_path, "locked", pad=2 * 1024 * 1024)
    assert census.run_store_census(tmp_path)["reported"] == 0
    os.chmod(target / "scratchpad", 0)
    try:
        result = census.run_store_census(tmp_path)
        assert result["reported"] == 0
        assert _unaccounted(tmp_path, census.UNACCOUNTED_CONTENT_POLICY) == []
        payload = json.loads((tmp_path / "sessions" / census.CENSUS_NAME).read_text())
        measured = payload["sessions"]["locked"]
        assert measured["contents_complete"] is False
        assert measured["skipped_entries"] >= 1
        assert payload["complete"] is False and payload["enumerated"] is True
    finally:
        os.chmod(target / "scratchpad", 0o700)


def test_the_recorded_offset_is_the_snapshot_the_diff_read(tmp_path: Path) -> None:
    """Review round 1, M2: report rows must not move the next pass's offset.

    The offset is the size the diff read BEFORE this pass appended anything.
    A post-append re-stat would skip every row that landed in between and a
    recorded removal could be reported as unaccounted on a later pass.
    """
    _session(tmp_path, "kept")
    gone = _session(tmp_path, "gone")
    assert census.run_store_census(tmp_path)["count"] == 2
    shutil.rmtree(gone)

    assert census.run_store_census(tmp_path)["reported"] == 1

    payload = json.loads((tmp_path / "sessions" / census.CENSUS_NAME).read_text())
    assert payload["log_bytes"] == 0, "the offset must be the pre-append snapshot"
    log_size = (tmp_path / "sessions" / CLEANUP_LOG_NAME).stat().st_size
    assert log_size > 0
    # The unaccounted row written by this pass sits at the offset, so the next
    # pass reads it and still does not duplicate the report.
    assert census.run_store_census(tmp_path)["reported"] == 0
    assert len(_unaccounted(tmp_path, census.UNACCOUNTED_REMOVAL_POLICY)) == 1


def test_an_interrupted_pass_keeps_the_previous_baseline(tmp_path: Path) -> None:
    """Review round 1, M3: a partial baseline must never evict a usable one."""
    _session(tmp_path, "k1")
    gone = _session(tmp_path, "k2")
    census.run_store_census(tmp_path)
    baseline = (tmp_path / "sessions" / census.CENSUS_NAME).read_text()

    stop = threading.Event()
    stop.set()
    interrupted = census.run_store_census(tmp_path, should_stop=stop.is_set)
    assert interrupted["enumerated"] is False
    assert (tmp_path / "sessions" / census.CENSUS_NAME).read_text() == baseline

    shutil.rmtree(gone)
    assert census.run_store_census(tmp_path)["reported"] == 1
    assert _unaccounted(tmp_path, census.UNACCOUNTED_REMOVAL_POLICY)[0]["session"] == "k2"


def test_an_interrupted_pass_warns_that_the_audit_did_not_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Review round 1, M4: a silent audit is the failure an auditor cannot see."""
    _session(tmp_path, "one")
    stop = threading.Event()
    stop.set()
    with caplog.at_level(logging.WARNING, logger=census.__name__):
        census.run_store_census(tmp_path, should_stop=stop.is_set)
    assert any("not fully enumerated" in record.message for record in caplog.records)


def test_a_shrunken_log_is_rescanned_from_zero(tmp_path: Path) -> None:
    """A rotated/truncated log must not be read from a stale offset."""
    _session(tmp_path, "kept")
    gone = _session(tmp_path, "gone")
    census.run_store_census(tmp_path)
    # Give the baseline a non-zero offset, then rotate the log away.
    from local_operator.session.cleanup import _append_cleanup_log

    assert _append_cleanup_log(tmp_path / "sessions", {"session": "other", "policy": "x"})
    census.run_store_census(tmp_path)
    (tmp_path / "sessions" / CLEANUP_LOG_NAME).write_text("", encoding="utf-8")
    shutil.rmtree(gone)

    result = census.run_store_census(tmp_path)

    assert result["enumerated"] is True
    assert result["reported"] == 1
    # The rotated log had nothing to read, so the pass's offset is 0 and its
    # own row sits AFTER it — the shape the next pass reads from.
    payload = json.loads((tmp_path / "sessions" / census.CENSUS_NAME).read_text())
    assert payload["log_bytes"] == 0
    assert (tmp_path / "sessions" / CLEANUP_LOG_NAME).stat().st_size > 0
    assert census.run_store_census(tmp_path)["reported"] == 0


def test_the_global_budget_stops_enumeration_without_evicting_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path, "a")
    _session(tmp_path, "b")
    census.run_store_census(tmp_path)
    baseline = (tmp_path / "sessions" / census.CENSUS_NAME).read_text()
    monkeypatch.setattr(census, "CENSUS_MAX_TOTAL_ENTRIES", 1)

    result = census.run_store_census(tmp_path)

    assert result["enumerated"] is False
    assert (tmp_path / "sessions" / census.CENSUS_NAME).read_text() == baseline


def test_a_capped_pad_marks_the_pass_incomplete_but_still_enumerates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path, "big", pad=4096)
    monkeypatch.setattr(census, "CENSUS_MAX_ENTRIES_PER_SESSION", 1)

    result = census.run_store_census(tmp_path)

    assert result["complete"] is False and result["enumerated"] is True
    payload = json.loads((tmp_path / "sessions" / census.CENSUS_NAME).read_text())
    assert payload["sessions"]["big"]["contents_complete"] is False


def test_usable_baseline_refuses_partial_or_unknown_documents() -> None:
    usable = {
        "schema": census.CENSUS_SCHEMA,
        "complete": True,
        "enumerated": True,
        "log_bytes": 0,
        "sessions": {},
    }
    assert census._usable_baseline(usable) is usable
    assert census._usable_baseline({**usable, "enumerated": False}) is None
    assert census._usable_baseline({**usable, "schema": 99}) is None
    assert census._usable_baseline({**usable, "log_bytes": -1}) is None
    # A pre-`enumerated` document falls back to `complete`.
    legacy = {k: v for k, v in usable.items() if k != "enumerated"}
    assert census._usable_baseline(legacy) is legacy

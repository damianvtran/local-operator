"""The durable ``video_job_v1`` record: shape, reader, wake vocabulary.

The reader runs against a REAL transcript file written by hand in the row
format ``Transcript.append_custom`` produces (``type: custom``, payload
``{custom_type, details}``), because the point of the reader is that it finds
those rows without parsing the journal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_operator.session import transcript as transcript_module
from local_operator.video import records as video_records


def _row(*, entry_id: str, ts: float, custom_type: str, details: dict[str, Any]) -> str:
    return json.dumps(
        {
            "id": entry_id,
            "ts": ts,
            "type": transcript_module.ENTRY_CUSTOM,
            "payload": {"custom_type": custom_type, "details": details},
        },
        separators=(",", ":"),
    )


def _write_rows(directory: Path, rows: list[str]) -> None:
    (directory / "transcript.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_record_shape_and_omission() -> None:
    record = video_records.build_record(
        job_id="abc123",
        state="running",
        provider="fal",
        route_label="FAL",
        model="minimax/h3-max/image-to-video",
        mode="i2v",
        params={"prompt": "p", "duration_s": 8},
        sources=[{"kind": "image", "attachment": "d" * 32}],
        provider_ref={"request_id": "r1"},
        at="2026-10-10T10:00:00+00:00",
    )
    assert record["job"] == "abc123"
    assert record["v"] == video_records.VIDEO_JOB_RECORD_VERSION == 1
    assert record["state"] == "running"
    assert record["at"] == "2026-10-10T10:00:00+00:00"
    # Omitted, never null: a partial row must not claim absence it did not
    # verify.
    assert "result" not in record
    assert "outcome_note" not in record
    assert record["params"] == {"prompt": "p", "duration_s": 8}


def test_build_record_rejects_unknown_state() -> None:
    import pytest

    with pytest.raises(ValueError):
        video_records.build_record(job_id="x", state="nonsense")


def test_newest_records_by_job_at_then_order() -> None:
    # Rows arrive NEWEST-FIRST (read_custom_entries walks backwards from the
    # end of the transcript), so on an identical `at` stamp the FIRST row is
    # the later write (round-1 QA blocker Q2: the registered/running pair
    # shares one second, and picking the older row dropped the provider_ref
    # the post-restart cold follow needs).
    rows = [
        {"job": "a", "at": "2026-10-10T10:05:00+00:00", "state": "failed"},
        {"job": "a", "at": "2026-10-10T10:05:00+00:00", "state": "completed"},
        {"job": "b", "at": "2026-10-10T10:01:00+00:00", "state": "running"},
        {"job": "a", "at": "2026-10-10T10:00:00+00:00", "state": "registered"},
    ]
    grouped = video_records.newest_records_by_job(rows)  # type: ignore[arg-type]
    assert grouped["a"]["state"] == "failed", "the first-seen same-second row is the newest"
    assert grouped["b"]["state"] == "running"
    # QA's deterministic repro, now going the right way: the newest row of the
    # tie carries the ref and must survive.
    tie = [
        {
            "job": "j",
            "at": "2026-10-10T20:14:30+00:00",
            "state": "running",
            "provider_ref": {"request_id": "01a12773-ebab-78a3-ad53-d50710b3e351"},
        },
        {"job": "j", "at": "2026-10-10T20:14:30+00:00", "state": "registered"},
    ]
    newest = video_records.newest_records_by_job(tie)  # type: ignore[arg-type]
    assert newest["j"]["state"] == "running"
    assert newest["j"].get("provider_ref"), "the same-second tie must not drop the ref"
    # A strictly newer stamp wins regardless of position in the list.
    assert (
        video_records.newest_records_by_job(
            [  # type: ignore[arg-type]
                {"job": "c", "at": "2026-10-10T10:00:00+00:00", "state": "registered"},
                {"job": "c", "at": "2026-10-10T10:01:00+00:00", "state": "completed"},
            ]
        )["c"]["state"]
        == "completed"
    )
    # Malformed rows are skipped individually, never fatal.
    grouped = video_records.newest_records_by_job(
        [{"nope": 1}, {"job": "", "at": "x"}]  # type: ignore[list-item]
    )
    assert grouped == {}


def test_read_video_records_roundtrip(tmp_path: Path) -> None:
    rows = [
        _row(
            entry_id="1",
            ts=1.0,
            custom_type=video_records.VIDEO_JOB_CUSTOM_TYPE,
            details={"job": "aaa", "at": "2026-10-10T10:00:00+00:00", "state": "registered"},
        ),
        _row(
            entry_id="2",
            ts=2.0,
            custom_type=video_records.VIDEO_JOB_CUSTOM_TYPE,
            details={
                "job": "aaa",
                "at": "2026-10-10T10:03:00+00:00",
                "state": "completed",
                "result": {"digest": "d" * 32},
            },
        ),
        # A different custom type must not leak into the map.
        _row(entry_id="3", ts=3.0, custom_type="wake_schedules", details={"job": "aaa"}),
        _row(
            entry_id="4",
            ts=4.0,
            custom_type=video_records.VIDEO_JOB_CUSTOM_TYPE,
            details={"job": "bbb", "at": "2026-10-10T10:02:00+00:00", "state": "failed"},
        ),
    ]
    _write_rows(tmp_path, rows)
    found = video_records.read_video_records(tmp_path)
    assert set(found) == {"aaa", "bbb"}
    assert found["aaa"]["state"] == "completed"
    assert found["aaa"]["result"]["digest"] == "d" * 32
    assert found["bbb"]["state"] == "failed"
    assert video_records.record_is_terminal(found["aaa"])
    assert not video_records.record_is_terminal(found.get("ccc", {"state": "running"}))


def test_read_video_records_missing_dir_answers_empty(tmp_path: Path) -> None:
    assert video_records.read_video_records(tmp_path / "nope") == {}
    assert video_records.read_video_records(None) == {}


def test_check_wake_vocabulary() -> None:
    assert video_records.check_wake_id("abc123") == "video-check-abc123"
    message = video_records.check_wake_message("abc123")
    assert "abc123" in message and "video_status" in message
    request = video_records.check_wake_request("abc123")
    assert request["notify"] is False
    assert request["in"] == "5m"


def test_bookkeeping_pin_matches_video_records() -> None:
    # The transcript names the type as a literal (it does not import the
    # writer); this pin is the drift gate, the ``stt.sidecar`` precedent.
    assert video_records.VIDEO_JOB_CUSTOM_TYPE in transcript_module.BOOKKEEPING_CUSTOM_TYPES

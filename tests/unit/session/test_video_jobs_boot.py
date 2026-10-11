"""The session-side video seams: the journal door and the boot conversion.

A non-terminal ``video_job_v1`` record that outlived its runtime is the
restart case (criterion b): ``async_init``'s ``_boot_video_jobs`` must ensure
the check-wake exists — and must be idempotent, because a boot path runs on
every open.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.session.test_session import ScriptedStream, make_session


def _record(job_id: str) -> dict[str, Any]:
    return {
        "job": job_id,
        "v": 1,
        "at": "2026-10-10T10:00:00+00:00",
        "state": "running",
        "provider": "fal",
        "model": "minimax/h3-max/image-to-video",
        "mode": "i2v",
        "params": {"prompt": "p"},
        "provider_ref": {"request_id": "r1"},
    }


@pytest.mark.asyncio
async def test_boot_video_jobs_ensures_check_wake_idempotently(tmp_path) -> None:
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        # The journal door first: the record lands as a custom row the boot
        # scan can find on disk.
        await session.journal_video_job(_record("vid1"))

        await session._boot_video_jobs()
        arming = [
            row for row in session._wake.schedules if getattr(row, "id", "") == "video-check-vid1"
        ]
        assert len(arming) == 1, "a pending record must get its check-wake"

        await session._boot_video_jobs()
        arming = [
            row for row in session._wake.schedules if getattr(row, "id", "") == "video-check-vid1"
        ]
        assert len(arming) == 1, "boot runs on every open; a second pass must not double-arm"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_boot_video_jobs_skips_terminal_records(tmp_path) -> None:
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        record = _record("vid2")
        record["state"] = "completed"
        await session.journal_video_job(record)
        await session._boot_video_jobs()
        assert not [
            row for row in session._wake.schedules if getattr(row, "id", "") == "video-check-vid2"
        ], "a settled job has nothing left to check"
    finally:
        await session.dispose()

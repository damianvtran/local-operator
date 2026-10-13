"""Opt-in Aida phase origins must retain shared writer identity/rollback gates."""
from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.aida import proactive
from local_operator.harness.wake_types import WakeSchedule
from local_operator.wakes import arm
from tests.unit.aida.conftest import isolated_root_path
from tests.unit.wakes.test_arm import _rows_from_transcript, _session, _snapshot_count

NOW = 1_800_000_000_000
HOUR = 3_600_000
SID = "cadence01"
FIRST = f"aida-cadence:v1:4:{NOW}:{NOW + 4 * HOUR}"
SECOND = f"aida-cadence:v1:6:{NOW}:{NOW + 6 * HOUR}"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    result = isolated_root_path(tmp_path, monkeypatch)
    monkeypatch.setattr(arm, "_ensure_supervisor", lambda *_: "test-no-install")
    return result


def seed(root: Path) -> Path:
    cadence = WakeSchedule(id=proactive.CADENCE_ID, message="first", created_at=NOW,
                           next_due_at=NOW + 4 * HOUR, request_id=FIRST)
    user = WakeSchedule(id="w1", message="independent", next_due_at=NOW + 10 * HOUR)
    return _session(root, SID, [user.model_dump(), cadence.model_dump()])


@pytest.mark.asyncio
async def test_same_origin_arm_retry_settles_but_different_occurrence_does_not(root: Path) -> None:
    directory = seed(root)
    before = _snapshot_count(directory)
    request = {"message": "first", "in": "4h"}
    outcome = await arm.arm_wake(
        root, SID, request, wake_id=proactive.CADENCE_ID, now_ms=NOW, request_id=FIRST,
    )
    assert outcome.wake_id == proactive.CADENCE_ID
    assert _snapshot_count(directory) == before
    with pytest.raises(arm.WakeWriteError, match="already exists"):
        await arm.arm_wake(root, SID, request, wake_id=proactive.CADENCE_ID,
                           now_ms=NOW, request_id=SECOND)
    assert _snapshot_count(directory) == before
    await arm.cancel_wake(root, SID, proactive.CADENCE_ID, now_ms=NOW)
    await arm.arm_wake(root, SID, {"message": "second", "in": "6h"},
                       wake_id=proactive.CADENCE_ID, now_ms=NOW, request_id=SECOND)
    rows = _rows_from_transcript(directory)
    assert [row["id"] for row in rows] == ["w1", proactive.CADENCE_ID]
    assert rows[1]["request_id"] == SECOND


@pytest.mark.asyncio
async def test_ordinary_edit_retains_origin_and_opt_in_edit_is_atomic_retryable(root: Path) -> None:
    directory = seed(root)
    original_user = _rows_from_transcript(directory)[0]
    await arm.edit_wake(root, SID, proactive.CADENCE_ID, {"message": "ordinary"}, now_ms=NOW)
    assert _rows_from_transcript(directory)[1]["request_id"] == FIRST
    patch = {"message": "new mode", "in": "6h"}
    await arm.edit_wake(root, SID, proactive.CADENCE_ID, patch, now_ms=NOW,
                        request_id=SECOND, expected_request_id=FIRST)
    rows = _rows_from_transcript(directory)
    assert rows[0] == original_user
    assert (rows[1]["request_id"], rows[1]["next_due_at"]) == (SECOND, NOW + 6 * HOUR)
    count = _snapshot_count(directory)
    await arm.edit_wake(root, SID, proactive.CADENCE_ID, patch, now_ms=NOW,
                        request_id=SECOND, expected_request_id=FIRST)
    assert _snapshot_count(directory) == count


@pytest.mark.asyncio
async def test_expected_origin_guard_refuses_an_intervening_arm(root: Path) -> None:
    directory = seed(root)
    before = _rows_from_transcript(directory)
    with pytest.raises(arm.WakeWriteError, match="origin changed"):
        await arm.edit_wake(root, SID, proactive.CADENCE_ID, {"in": "6h"}, now_ms=NOW,
                            request_id=SECOND, expected_request_id="stale-arm")
    assert _rows_from_transcript(directory) == before


@pytest.mark.asyncio
async def test_failed_opt_in_edit_preserves_original(
    root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = seed(root)
    before = _rows_from_transcript(directory)

    async def fail_append(*_: object) -> str:
        raise OSError("injected append failure")

    monkeypatch.setattr(arm, "_append", fail_append)
    with pytest.raises(OSError, match="injected append failure"):
        await arm.edit_wake(root, SID, proactive.CADENCE_ID, {"in": "6h"}, now_ms=NOW,
                            request_id=SECOND, expected_request_id=FIRST)
    assert _rows_from_transcript(directory) == before


@pytest.mark.asyncio
async def test_raced_optional_origin_edit_rollback_keeps_another_same_id_arm(
    root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = seed(root)
    user = _rows_from_transcript(directory)[0]
    append = arm._append
    raced = False
    peer = WakeSchedule(id=proactive.CADENCE_ID, message="peer replacement", created_at=NOW + 1,
                        next_due_at=NOW + 5 * HOUR, request_id="peer-distinct-origin")

    async def append_then_peer(target: Path, rows: list[WakeSchedule]) -> str:
        nonlocal raced
        entry = await append(target, rows)
        if not raced:
            raced = True
            await append(target, [WakeSchedule.model_validate(user), peer])
        return entry

    async def refuse_after_peer(*_: object) -> None:
        if raced:
            raise arm.WakeWriteError(
                "injected owner refusal", status=503, code="wake_owner_present",
            )

    monkeypatch.setattr(arm, "_append", append_then_peer)
    monkeypatch.setattr(arm, "_refuse_if_owned", refuse_after_peer)
    with pytest.raises(arm.WakeWriteError, match="injected owner refusal"):
        await arm.edit_wake(root, SID, proactive.CADENCE_ID, {"in": "6h"}, now_ms=NOW,
                            request_id=SECOND, expected_request_id=FIRST)
    assert _rows_from_transcript(directory) == [user, peer.model_dump()]

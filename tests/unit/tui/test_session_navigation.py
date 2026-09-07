"""Navigation generations protect identity and bound speculative resources."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.tui.session_navigation import SessionNavigation


@pytest.mark.asyncio
async def test_rapid_selection_prepares_only_latest_after_retired_work_settles():
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    prepared = []
    committed = []
    released = []
    pending = []
    active = 0
    peak = 0

    async def prepare(session_id):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        prepared.append(session_id)
        try:
            if session_id == "first":
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    # A filesystem read may finish despite cancellation. The
                    # generation still forbids it from becoming the input target.
                    await release_first.wait()
            return session_id
        finally:
            active -= 1

    async def release(value):
        released.append(value)

    navigation = SessionNavigation(
        prepare=prepare,
        commit=lambda session_id, value, generation: committed.append(value),
        release=release,
        pending=pending.append,
        failed=lambda session_id, error: pytest.fail(str(error)),
    )
    first = navigation.select("first")
    await asyncio.wait_for(first_started.wait(), 5)
    skipped = navigation.select("skipped")
    final = navigation.select("final")
    release_first.set()
    await asyncio.wait_for(final, 5)
    await asyncio.gather(first, skipped, return_exceptions=True)
    assert prepared == ["first", "final"]
    assert committed == ["final"]
    assert released == ["first"]
    assert peak == 1
    assert pending == ["first", "skipped", "final", ""]
    assert navigation.committed_id == "final"
    await navigation.close()
    assert not navigation._tasks


@pytest.mark.asyncio
async def test_failure_preserves_committed_identity_and_releases_input_boundary():
    errors = []
    pending = []

    async def prepare(session_id):
        raise ConnectionError("owner unavailable")

    async def release(value):
        pytest.fail("no resource was prepared")

    navigation = SessionNavigation(
        prepare=prepare,
        commit=lambda *_: pytest.fail("failed preparation committed"),
        release=release,
        pending=pending.append,
        failed=lambda session_id, error: errors.append((session_id, str(error))),
    )
    navigation.committed_id = "original"
    await navigation.select("unavailable")
    assert navigation.committed_id == "original"
    assert not navigation.requested_id
    assert pending == ["unavailable", ""]
    assert errors == [("unavailable", "owner unavailable")]
    await navigation.close()

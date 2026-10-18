"""A cancel before the first committed row leaves no empty target session."""

from __future__ import annotations

import pytest

from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.types import ImportRequest

SOURCE = "omp"
SESSION_ID = "omp-synthetic"


@pytest.mark.asyncio
async def test_cancel_before_first_batch_removes_the_empty_target(omp_store):
    _home, config = omp_store
    service = HarnessImportService(config)
    cancelled = False

    def progress(state):
        nonlocal cancelled
        if state.phase in ("session", "selecting") and not cancelled:
            cancelled = True
            service.cancel()

    result = await service.import_sessions(
        ImportRequest(sources=[SOURCE], session_ids={SOURCE: [SESSION_ID]}),
        progress=progress,
    )

    assert result.state == "cancelled"
    sessions = config / "sessions"
    assert not sessions.exists() or not list(
        sessions.iterdir()
    ), "a cancelled import left an empty target session"


@pytest.mark.asyncio
async def test_non_cancel_failure_also_cleans_the_empty_target(omp_store, monkeypatch):
    """Any first-batch failure leaves no session directory, not just cancel."""
    from local_operator.harness_import import manifest as manifest_store

    _home, config = omp_store
    service = HarnessImportService(config)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(manifest_store, "write_manifest", boom)
    result = await service.import_sessions(
        ImportRequest(sources=[SOURCE], session_ids={SOURCE: [SESSION_ID]})
    )

    assert result.sessions[0].outcome == "failed"
    sessions = config / "sessions"
    assert not sessions.exists() or not list(
        sessions.iterdir()
    ), "a failed import left an empty target session"

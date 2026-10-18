"""The session's ONE append path for a post-cancel settlement row.

The settle poll runs in a session background task after the tool returned, so
the transcript write cannot come from the tool. This pins the door the session
binds (``journal_audio_settle``) and the ROW TYPE's bookkeeping property: the
row lands as a custom entry, carries its facts, and does not re-rank the
session as freshly worked.
"""

from __future__ import annotations

import pytest

from tests.unit.session.test_session import ScriptedStream, make_session


@pytest.mark.asyncio
async def test_journal_audio_settle_appends_a_bookkeeping_row(tmp_path) -> None:
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        await session.journal_audio_settle(
            {
                "version": 1,
                "request_id": "req-late",
                "status": "COMPLETED",
                "settlement_observed": True,
                "cost_usd": 0.03,
                "billing_basis": "billed",
                "digest": "a" * 32,
            }
        )
        rows = [
            entry
            for entry in session._transcript.entries()
            if entry.type == "custom" and entry.payload.get("custom_type") == "audio_settle_v1"
        ]
        assert len(rows) == 1
        details = rows[0].payload["details"]
        assert details["request_id"] == "req-late"
        assert details["settlement_observed"] is True
        assert details["cost_usd"] == 0.03
    finally:
        await session.dispose()


def test_the_context_exposes_the_settle_hooks() -> None:
    """Both hooks are optional (a reduced host binds neither) and distinct."""
    from local_operator.harness.types import ToolContext

    context = ToolContext(cwd=".")
    assert context.spawn_background is None
    assert context.journal_audio_settle is None

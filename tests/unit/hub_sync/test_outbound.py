"""Paired A/P projection: local edits kept, remote-only content preserved, no-op detection."""

from __future__ import annotations

from local_operator.hub_sync.merge import MergeOptions
from local_operator.hub_sync.outbound import project_outbound

A = {"instructions": "# A\nbase rule\n", "description": "d"}
P = {"instructions": "# A\nbase rule\n", "description": "d"}


def test_local_edit_since_capture_is_projected() -> None:
    local = {"instructions": "# A\nbase rule\nMY EDIT\n", "description": "d"}
    result = project_outbound(
        "agent",
        captured_local=A,
        accepted=P,
        current_local=local,
        remote_latest=P,
        opts=MergeOptions(allow_llm=False),
    )
    assert not result.no_op
    assert "MY EDIT" in result.fields["instructions"]


def test_remote_only_content_in_p_is_preserved() -> None:
    # P gained a remote-only section the local copy never had.
    P2 = {"instructions": "# A\nbase rule\nREMOTE ONLY\n", "description": "d"}
    result = project_outbound(
        "agent",
        captured_local=A,
        accepted=P2,
        current_local=A,
        remote_latest=P2,
        opts=MergeOptions(allow_llm=False),
    )
    assert "REMOTE ONLY" in result.fields["instructions"]


def test_identical_remote_is_a_no_op() -> None:
    result = project_outbound(
        "agent",
        captured_local=A,
        accepted=P,
        current_local=A,
        remote_latest=P,
        opts=MergeOptions(allow_llm=False),
    )
    assert result.no_op

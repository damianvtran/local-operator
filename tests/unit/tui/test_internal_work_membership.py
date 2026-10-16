"""Owned-work policy; separate from the shared quiet-group parity corpus.

These tests exercise pure classification, not replay/widget admission or focus.
Those adapters require their own real-path evidence before this feature ships.
"""

from local_operator.tui.internal_work_membership import (
    WorkRecord,
    project_internal_work,
)


def _user(source_id: str = "user-a") -> WorkRecord:
    return WorkRecord(source_id, "user")


def _tool(
    source_id: str, *, launched_job_id: str | None = None, material: bool = False
) -> WorkRecord:
    return WorkRecord(
        source_id, "tool", tool_name="bash", launched_job_id=launched_job_id, material=material
    )


def _job(
    source_id: str = "job", *, job_id: str | None = "job-a", status: str | None = None
) -> WorkRecord:
    return WorkRecord(
        source_id,
        "custom",
        custom_type="job_result",
        producer="job_result",
        job_id=job_id,
        status=status,
        level="info",
    )


def test_never_sent_tool_cannot_authorise_receipts_in_replay_or_incremental_projection() -> None:
    from dataclasses import replace

    from local_operator.tui.internal_work_membership import WorkProjector

    for never_sent in (True, False):
        records = [_user(), replace(_tool("tool"), never_sent=never_sent), _job()]
        replay = project_internal_work(records)
        live = WorkProjector()
        incremental = {row.source_id: live.step(row) for row in records}
        expected = "boundary" if never_sent else "receipt"
        assert replay["job"].role == incremental["job"].role == expected
        assert replay["job"].owner_evidence == incremental["job"].owner_evidence
        if never_sent:
            assert replay["job"].owner_evidence == "no-prior-tool"


def test_owned_interval_preserves_order_and_separates_bridges_from_receipts() -> None:
    records = [
        _user(),
        _tool("shell"),
        _job(),
        WorkRecord(
            "hub",
            "custom",
            custom_type="hub_message",
            direction="to_parent",
            expects_reply=False,
            steer=False,
            job_id="job-a",
        ),
        WorkRecord("wait", "tool", tool_name="wait"),
        WorkRecord("compact", "compaction"),
        WorkRecord(
            "completed",
            "notice",
            producer="subagent_completion",
            job_id="job-a",
            status="completed",
            level="info",
        ),
        WorkRecord("send", "tool", tool_name="send"),
        _tool("tail"),
        WorkRecord("final", "assistant", final=True),
    ]
    projected = project_internal_work(records)
    assert [projected[r.source_id].stretch_id for r in records[1:-1]] == ["shell"] * 8
    assert [projected[r.source_id].role for r in records[1:-1]] == [
        "action",
        "receipt",
        "receipt",
        "action",
        "bridge",
        "bridge",
        "action",
        "action",
    ]
    assert projected["job"].outcome == "unknown"
    assert projected["final"].role == "boundary"
    assert projected["user-a"].role == "boundary"


def test_missing_opener_or_missing_preceding_tool_never_grants_ownership() -> None:
    for records in [[_job()], [_tool("t"), _job()], [_user(), _job()]]:
        assert project_internal_work(records)["job"].role == "boundary"


def test_a_known_conflicting_job_owner_is_refused_but_a_launch_is_not_mandatory() -> None:
    assert project_internal_work([_user(), _tool("t"), _job()])["job"].role == "receipt"
    records = [
        _user("old"),
        _tool("launch", launched_job_id="job-a"),
        WorkRecord("final", "assistant", final=True),
        _user("new"),
        _tool("t"),
        _job(),
    ]
    member = project_internal_work(records)["job"]
    assert member.role == "boundary"
    assert member.owner_evidence == "conflicting-job-owner"


def test_unknown_completion_malformed_result_and_typed_failure_keep_attention_visible() -> None:
    for row in [
        _job(job_id=" "),
        _job(status="failed"),
        WorkRecord(
            "job",
            "notice",
            producer="subagent_completion",
            job_id="job-a",
            status="cancelled",
            level="info",
        ),
    ]:
        assert project_internal_work([_user(), _tool("t"), row])["job"].role == "boundary"


def test_real_final_and_steer_do_not_inherit_tool_evidence() -> None:
    assert (
        project_internal_work(
            [_user(), _tool("t"), WorkRecord("final", "assistant", final=True), _job()]
        )["job"].scope_id
        is None
    )
    assert (
        project_internal_work([_user(), _tool("t"), _user("steer"), _job()])["job"].owner_evidence
        == "no-prior-tool"
    )


def test_material_and_first_token_split_presentation_without_erasing_the_scope() -> None:
    for boundary in [
        _tool("boundary", material=True),
        WorkRecord("boundary", "assistant", streaming=True),
    ]:
        rows = [_user(), _tool("t"), boundary, _tool("after"), _job()]
        projected = project_internal_work(rows)
        assert projected["boundary"].role == "boundary"
        assert projected["after"].stretch_id == "after"
        assert projected["job"].stretch_id == "after"

"""End-to-end outbound push against a mock server: no-op, CAS, timeout, fencing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from local_operator.hub_sync.budget import OperationBudget
from local_operator.hub_sync.merge import MergeOptions
from local_operator.hub_sync.outbox import Outbox
from local_operator.hub_sync.push import (
    PushOutcome,
    RemoteRevisionChanged,
    push_item,
    receipt_for,
)

A = {"instructions": "# A\nbase rule\n", "description": "d"}


class MockServer:
    """A preview/commit server: monotonic revisions + optional 412/timeout."""

    def __init__(self, *, revision: str = "r1", conflict: bool = False, hang: bool = False) -> None:
        self.revision = revision
        self.conflict = conflict
        self.hang = hang
        self.committed: list[Mapping[str, Any]] = []

    async def preview(self, *, kind, payload, budget):
        if self.hang:
            from local_operator.hub_sync.budget import OperationDeadlineExceeded

            raise OperationDeadlineExceeded("preview exceeded")
        return {"content_revision": self.revision, "operation_id": "srv-op"}

    async def commit(self, *, kind, payload, expected_revision, operation_id, budget):
        if self.conflict:
            raise RemoteRevisionChanged("remote-revision-changed")
        self.committed.append(payload)
        return {"content_revision": "r2", "operation_id": operation_id}


def _budget() -> OperationBudget:
    return OperationBudget.start("automatic", "op", 1)


async def _run(tmp_path: Path, **kw) -> tuple[PushOutcome, Outbox, MockServer]:
    server = MockServer(**kw)
    out = Outbox(tmp_path)
    outcome = await push_item(
        kind="agent",
        local_id="a1",
        captured_local=A,
        accepted=A,
        current_local={"instructions": "# A\nbase rule\nNEW\n", "description": "d"},
        remote_latest=A,
        transport=server,
        outbox=out,
        operation_id="op1",
        budget=_budget(),
        opts=MergeOptions(allow_llm=False),
    )
    return outcome, out, server


def test_a_local_edit_is_previewed_then_committed(tmp_path: Path) -> None:
    import asyncio

    outcome, out, server = asyncio.run(_run(tmp_path))
    assert outcome.state == "pushed" and outcome.revision == "r2"
    assert server.committed and "NEW" in server.committed[0]["instructions"]
    assert out.read("op1").state == "done"


def test_identical_remote_settles_a_no_op_without_paying(tmp_path: Path) -> None:
    import asyncio

    server = MockServer()
    out = Outbox(tmp_path)
    outcome = asyncio.run(
        push_item(
            kind="agent",
            local_id="a1",
            captured_local=A,
            accepted=A,
            current_local=A,
            remote_latest=A,
            transport=server,
            outbox=out,
            operation_id="op1",
            budget=_budget(),
            opts=MergeOptions(allow_llm=False),
        )
    )
    assert outcome.state == "no-op"
    assert server.committed == []  # no preview/commit reached the server


def test_a_cas_conflict_refuses_to_resubmit(tmp_path: Path) -> None:
    import asyncio

    outcome, out, server = asyncio.run(_run(tmp_path, conflict=True))
    assert outcome.state == "conflict" and outcome.reason == "remote-revision-changed"
    assert server.committed == []  # the stale body was NOT resubmitted
    assert out.read("op1").state == "failed"


def test_an_aborted_dispatch_is_pending_reconciliation_not_failed(tmp_path: Path) -> None:
    import asyncio

    outcome, out, _ = asyncio.run(_run(tmp_path, hang=True))
    assert outcome.state == "pending-reconciliation"
    assert out.read("op1").state == "pending-reconciliation"  # replayable, not failed


def test_no_outbound_receipt_refuses_without_publishing(tmp_path: Path) -> None:
    import asyncio

    server = MockServer()
    out = Outbox(tmp_path)
    outcome = asyncio.run(
        push_item(
            kind="agent",
            local_id="a1",
            captured_local=None,
            accepted=None,
            current_local=A,
            remote_latest=A,
            transport=server,
            outbox=out,
            operation_id="op1",
            budget=_budget(),
            opts=MergeOptions(allow_llm=False),
        )
    )
    assert outcome.state == "refused" and outcome.reason == "no-outbound-receipt"
    assert server.committed == []


def test_a_stale_generation_cannot_settle_after_a_new_attempt(tmp_path: Path) -> None:
    out = Outbox(tmp_path)
    out.enqueue(operation_id="op1", kind="agent", local_id="a1", direction="push", generation=1)
    out.bump_generation("op1")  # a new attempt starts
    # The late worker from generation 1 is fenced out of settlement.
    assert out.settle("op1", generation=1, result={"revision": "r9"}) is False
    assert out.read("op1").state == "pending"


def test_receipt_for_pairs_the_sent_local_with_the_accepted_remote() -> None:
    accepted = {"instructions": "# A\ngeneralized\n", "content_revision": "r2", "operation_id": "s"}
    receipt = receipt_for(
        "agent", accepted, {"instructions": "# A\nbase rule\nNEW\n", "description": "d"}
    )
    assert receipt.local_fields["instructions"].endswith("NEW\n")
    assert receipt.remote_revision == "r2"

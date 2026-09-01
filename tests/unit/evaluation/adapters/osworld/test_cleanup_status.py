"""Cleanup status mapping: an unconfirmed teardown must keep rescue_required.

This is the orphaned-instance safety net asserted end to end through the REAL
``aggregate_cleanup``, not just against the adapter's return value. The
distinction matters because ``not_needed`` is treated as CLEAN by the
aggregate: an adapter that reports it for a teardown it could not confirm
silently retires the rescue obligation, and on the AWS path that is an EC2
instance billing forever.

Round-1 review finding F3: the mapping sent ``terminate-unconfirmed`` and
``terminate-denied`` to ``not_needed``, which cleared ``rescue_required``
while the docstring claimed the opposite. These tests pin the corrected
behaviour so it cannot regress once PR 2 makes those codes reachable.
"""

from __future__ import annotations

import pytest
from lop_osworld_v2_adapter import cleanup
from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter
from lop_osworld_v2_adapter.providers.fake import FakeProvider

from local_operator.evaluation.lifecycle import aggregate_cleanup, record_cleanup

_EPISODE = "ep-cleanup-status"


class _CodedProvider(FakeProvider):
    """A provider that returns one exact teardown code, including the codes the
    fake never produces on its own but a real AWS provider will."""

    def __init__(self, code: str) -> None:
        super().__init__()
        self._code = code

    async def terminate(self, instance_ref: str) -> str:
        return self._code


async def _outcome_for(code: str) -> tuple[str, str]:
    """Run the release-instance action against a provider returning ``code``."""

    adapter = OSWorldV2Adapter(provider_factory=lambda: _CodedProvider(code))
    adapter._provider = _CodedProvider(code)
    status, evidence, _duration = await adapter._run_cleanup_action(
        "release_instance", f"lop-ep-{_EPISODE}"
    )
    return status, evidence


def _rescue_required(status: str, evidence: str) -> bool:
    """Ask the REAL aggregate whether this outcome retires the obligation."""

    refs = cleanup.CleanupRefs.mint(_EPISODE)
    plan = cleanup.build_cleanup_plan(_EPISODE, refs)
    receipts = tuple(
        record_cleanup(
            plan,
            action.action_id,
            # Only the release-instance action carries the status under test;
            # the other two succeed so the aggregate's exact-coverage rule is
            # satisfied and the flag reflects this action alone.
            status=(  # type: ignore[arg-type]
                status if action.kind == "release_instance" else "succeeded"
            ),
            evidence_code=evidence if action.kind == "release_instance" else "session-closed",
            duration_ms=1,
        )
        for action in plan.actions
    )
    return aggregate_cleanup(plan, receipts).rescue_required


@pytest.mark.asyncio
async def test_confirmed_termination_succeeds_and_clears_rescue() -> None:
    status, evidence = await _outcome_for(cleanup.EVIDENCE_INSTANCE_TERMINATED)
    assert status == "succeeded"
    assert _rescue_required(status, evidence) is False


@pytest.mark.asyncio
async def test_absent_instance_is_not_needed_and_clears_rescue() -> None:
    # We looked and there was nothing there: the only honest "clean" case
    # besides a confirmed terminate.
    status, evidence = await _outcome_for(cleanup.EVIDENCE_INSTANCE_ABSENT)
    assert status == "not_needed"
    assert _rescue_required(status, evidence) is False


@pytest.mark.asyncio
async def test_unconfirmed_termination_preserves_rescue_required() -> None:
    status, evidence = await _outcome_for(cleanup.EVIDENCE_TERMINATE_UNCONFIRMED)
    assert status == "attempted"
    assert _rescue_required(status, evidence) is True


@pytest.mark.asyncio
async def test_denied_termination_preserves_rescue_required() -> None:
    status, evidence = await _outcome_for(cleanup.EVIDENCE_TERMINATE_DENIED)
    assert status == "attempted"
    assert _rescue_required(status, evidence) is True


@pytest.mark.asyncio
async def test_an_unknown_code_fails_safe_toward_rescue() -> None:
    # A future provider inventing a code must degrade toward a redundant
    # rescue, never toward a silent leak.
    status, evidence = await _outcome_for("some-future-code")
    assert status == "attempted"
    assert _rescue_required(status, evidence) is True


@pytest.mark.asyncio
async def test_a_rescue_worker_without_a_provider_cannot_claim_the_instance_is_gone() -> None:
    """No provider means we could not LOOK, which is not the same as absent."""

    adapter = OSWorldV2Adapter()
    assert adapter._provider is None
    status, evidence, _ = await adapter._run_cleanup_action(
        "release_instance", f"lop-ep-{_EPISODE}"
    )
    assert status == "attempted"
    assert _rescue_required(status, evidence) is True


@pytest.mark.asyncio
async def test_an_unsupported_action_kind_preserves_rescue_required() -> None:
    """Round-2 review F7: unknown KINDS are the same hazard as unknown CODES.

    ``CleanupActionKind`` is a six-member Literal, so PR 2 adding
    ``delete_volume`` or ``restore_snapshot`` makes this reachable — and during
    rescue the plan arrives from a PERSISTED descriptor that may have been
    authored by a different adapter build than the one executing it. Reporting
    ``not_needed`` would assert "nothing to do" about a resource this build
    cannot name, clearing the obligation for something nobody released.
    """

    adapter = OSWorldV2Adapter()
    for kind in ("delete_volume", "restore_snapshot", "some_future_kind"):
        status, evidence, _ = await adapter._run_cleanup_action(kind, "lop-vol-ep-1")
        assert status == "attempted", kind
        assert evidence == cleanup.EVIDENCE_KIND_UNSUPPORTED, kind
        assert _rescue_required(status, evidence) is True, kind

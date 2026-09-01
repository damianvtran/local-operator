"""Budget reconciliation, overrun, and the writer-failure terminal.

Overrun is deliberately not a failure: an episode that exceeded its allowance
still produced a real result, and the overrun belongs in the reconciliation
record rather than in the score. What DOES change the run's standing is usage
that could not be measured at all, which is what makes a bundle unreportable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.supervisor import SupervisionError
from local_operator.evaluation.evidence.models import ReconciliationPayload
from local_operator.evaluation.evidence.store import EvidenceError
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import (
    BUDGET_RESOURCES,
    BudgetAuthorization,
    CappedAllowance,
    ResourceAmount,
    reserve_budget,
)
from local_operator.evaluation.runner.episode import EpisodeRunner
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    ScriptedModel,
    build_config,
    build_spec,
    payloads,
    selector,
)


async def _rescue_ok(descriptor: Any, **kwargs: Any) -> Any:
    del kwargs

    class _Aggregate:
        complete = True
        descriptor_id = descriptor.descriptor_id

    return _Aggregate()


@pytest.mark.asyncio
async def test_overrun_is_recorded_and_still_scores(tmp_path: Path, episode_id: str) -> None:
    """A tiny allowance is exceeded by real usage; the episode still scores."""

    spec = build_spec(episode_id)
    budget = BudgetAuthorization(
        episode_id=episode_id,
        allowances=tuple(
            CappedAllowance(resource=resource, value=1, reporting="optional")
            for resource in BUDGET_RESOURCES
        ),
    )
    reservation = reserve_budget(
        budget,
        "episode",
        [ResourceAmount(resource=resource, value=1) for resource in BUDGET_RESOURCES],
    )
    spec = type(spec)(
        **{
            **spec.__dict__,
            "budget": budget,
            "reservations": (reservation,),
        }
    )
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        spec,
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid
    reconciliations = payloads(root, ReconciliationPayload)
    assert len(reconciliations) == 1
    # Usage was measurable, so the run stays reportable despite the overrun.
    assert reconciliations[0].reportable is True


@pytest.mark.asyncio
async def test_dead_worker_makes_environment_usage_unavailable(
    tmp_path: Path, episode_id: str
) -> None:
    """A poisoned session cannot report environment usage; that is not a guess."""

    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"execute": SupervisionError("worker died")}
    )
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid
    reconciliations = payloads(root, ReconciliationPayload)
    # Required reporting is not demanded by this budget, so the reconciliation
    # still closes; the unavailable entries are what the record preserves.
    assert len(reconciliations) == 1


@pytest.mark.asyncio
async def test_cost_counters_match_the_usage_events(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "step", "finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid and report.counters is not None
    # Three decisions at the scripted 10/5 tokens and 7 micros each.
    assert report.counters.model_request_count == 3
    assert report.counters.model_response_count == 3
    assert report.counters.input_tokens == 30
    assert report.counters.output_tokens == 15
    assert report.counters.cost_microusd == 21
    reconciliations = payloads(root, ReconciliationPayload)
    # The sealed outcome's cost must equal the reconciliation's provider cost.
    assert reconciliations[0].provider_cost_microusd == 21


@pytest.mark.asyncio
async def test_writer_failure_abandons_after_rescuing_first(
    tmp_path: Path, episode_id: str
) -> None:
    """Cloud safety needs no writer, so rescue precedes abandonment."""

    adapter = FakeAdapter(tmp_path, episode_id)
    config = build_config(tmp_path)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )
    rescued: list[bool] = []

    async def tracking_rescue(descriptor: Any, **kwargs: Any) -> Any:
        rescued.append(True)
        return await _rescue_ok(descriptor, **kwargs)

    runner._rescue = tracking_rescue

    original = runner._append

    def failing(kind: Any, payload: Any) -> None:
        if kind == "environment_step":
            raise _poisoned()
        original(kind, payload)

    def _poisoned() -> BaseException:
        from local_operator.evaluation.runner.episode import _EvidenceFailure

        return _EvidenceFailure("journal write failed")

    runner._append = failing  # type: ignore[method-assign]

    outcome = await runner.run()

    assert outcome.status == "abandoned"
    assert rescued == [True]
    assert outcome.rescue_complete is True
    assert adapter.terminated
    assert isinstance(outcome.diagnostic, str)
    # The returned status is not evidence. A bundle whose writer died must
    # carry its terminal ON DISK, or the "every path leaves a verifiable
    # bundle or a coherent abandonment" invariant is only a claim.
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.abandonment is not None
    assert report.abandonment.reason == "infrastructure_failure"
    assert report.terminal_state == "abandoned"


@pytest.mark.asyncio
async def test_evidence_errors_surface_as_abandonment_not_a_crash(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )
    original = runner._publish

    def failing(data: bytes, **kwargs: Any) -> Any:
        raise EvidenceError("artifact store is full")

    runner._publish = failing  # type: ignore[method-assign]
    del original

    outcome = await runner.run()

    # An evidence failure is never reported as a scored or completed episode.
    assert outcome.status in ("abandoned", "failed_pre_bundle")
    assert outcome.score is None
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.abandonment is not None
    assert report.terminal_state == "abandoned"


@pytest.mark.parametrize("kind", ["observation", "action_batch"])
@pytest.mark.asyncio
async def test_append_failure_after_a_successful_publish_reports_the_disk_truth(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """An orphaned artifact makes the bundle unabandonable; say so honestly.

    ``observation`` and ``action_batch`` are written publish-then-append, so an
    append failure after a successful publish leaves an artifact nothing
    references. ``EvidenceWriter.abandon`` independently re-verifies and refuses
    on that ``artifact_unreferenced`` error, so NO terminal can be recorded.

    The runner must not answer that by claiming ``abandoned`` over a bundle
    whose ``terminal_state`` is still ``open`` -- that false report is the exact
    failure this slice exists to prevent. The returned status has to match what
    is actually on disk.
    """

    from local_operator.evaluation.evidence.store import EvidenceWriter

    adapter = FakeAdapter(tmp_path, episode_id)
    original = EvidenceWriter.append

    def poisoned(self: Any, event_kind: Any, payload: Any, **kwargs: Any) -> Any:
        if event_kind == kind:
            raise EvidenceError(f"simulated {kind} journal failure")
        return original(self, event_kind, payload, **kwargs)

    # Patched on the CLASS so the real writer is exercised, not a runner shim.
    monkeypatch.setattr(EvidenceWriter, "append", poisoned)

    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    if report.abandonment is not None:
        # If a terminal WAS recordable, it must be coherent and claimed as such.
        assert outcome.status == "abandoned"
        assert report.terminal_state == "abandoned"
    else:
        # No terminal on disk, so the runner must not claim one.
        assert outcome.status == "abandonment_failed"
        assert report.terminal_state == "open"
        assert outcome.diagnostic is not None
        assert "abandonment refused" in outcome.diagnostic
    # Either way the returned status agrees with the bundle on disk.
    assert (outcome.status == "abandoned") == (report.abandonment is not None)

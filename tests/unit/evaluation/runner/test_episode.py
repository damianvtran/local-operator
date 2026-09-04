"""Event mapping and terminal coherence for one episode.

Every test here drives the REAL ``VerifiedAdapterSession``, the REAL lifecycle
authorities and a REAL ``EvidenceWriter``; only the subprocess boundary is
faked. The standing assertion is that each terminal path leaves a bundle the
independent verifier accepts, because a runner that writes evidence no verifier
will take is indistinguishable from one that writes none.
"""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import Requirement, ScopedInfraValue
from local_operator.evaluation.adapters.supervisor import SupervisionError
from local_operator.evaluation.evidence.models import (
    CleanupPayload,
    EnvironmentStepPayload,
    ErrorPayload,
    ObservationPayload,
    ScoreArtifact,
)
from local_operator.evaluation.evidence.store import EvidenceWriter
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from local_operator.evaluation.runner.episode import (
    DISCLOSED_INFRA_METADATA_KEYS,
    EpisodeRunner,
)
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    RecordingResponder,
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


def _runner(
    tmp_path: Path,
    episode_id: str,
    *,
    adapter: FakeAdapter,
    model: ScriptedModel,
    responder: Any = None,
    max_steps: int = 4,
) -> EpisodeRunner:
    return EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=max_steps),
        selector=selector(tmp_path),
        model=model,
        responder=responder,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )


def _kinds(root: Path) -> list[str]:
    return [event.kind for event in verify_bundle(root).events]


@pytest.mark.asyncio
async def test_happy_episode_writes_every_event_in_protocol_order(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.reportability_label == "reportable"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert _kinds(root) == [
        # Preflight is event #0 and the commitment precedes all execution.
        "preflight",
        "budget_commitment",
        "lifecycle_transition",
        "observation",
        "model_request",
        "model_response",
        "usage_cost",
        "action_batch",
        "environment_step",
        "observation",
        "model_request",
        "model_response",
        "usage_cost",
        "action_batch",
        "finalization_start",
        "scoring_start",
        "scoring_result",
        "reconciliation",
        "cleanup",
        "lifecycle_transition",
    ]
    assert adapter.calls[:5] == [
        "handshake",
        "inspect_requirements",
        "prepare",
        "reset_start",
        "observe",
    ]


@pytest.mark.asyncio
async def test_step_event_precedes_its_output_observation(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    kinds = _kinds(root)
    step = kinds.index("environment_step")
    # The observation an execution produced must follow the step that declared
    # it; reversing them leaves the observation unbound to any receipt.
    assert kinds[step + 1] == "observation"


@pytest.mark.asyncio
async def test_mid_episode_observe_writes_no_duplicate_observation(
    tmp_path: Path, episode_id: str
) -> None:
    """``observe`` is a snapshot check, and a snapshot is not new evidence."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    observations = [payload.sequence for payload in payloads(root, ObservationPayload)]
    # One initial plus exactly one per executed step, with no repeats.
    assert observations == [0, 1]


@pytest.mark.asyncio
async def test_truncation_scores_normally_and_marks_the_last_step(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["step"] * 6),
        max_steps=2,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert len(steps) == 2
    assert [step.truncated for step in steps] == [False, True]


@pytest.mark.asyncio
async def test_ask_user_exchange_precedes_the_executed_ask_batch(
    tmp_path: Path, episode_id: str
) -> None:
    """Verify's one-batch-per-observation rule forces this choreography."""

    adapter = FakeAdapter(tmp_path, episode_id)
    responder = RecordingResponder("do the thing")
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["ask", "finish"]),
        responder=responder,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert responder.prompts == ["What next?"]
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    kinds = _kinds(root)
    exchange = kinds.index("user_simulator_exchange")
    assert kinds[exchange + 1] == "action_batch"
    assert kinds[exchange + 2] == "environment_step"
    assert "ask_user_exchange" in adapter.calls


@pytest.mark.asyncio
async def test_unanswered_ask_cancels_rather_than_leaving_it_open(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["ask", "finish"]),
        responder=RecordingResponder(None),
    )

    outcome = await runner.run()

    assert outcome.status == "cancelled"
    assert outcome.reportability_label == "cancelled"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    assert "cancel" in _kinds(root)
    assert outcome.score is not None and outcome.score.reason == "cancelled"


@pytest.mark.asyncio
async def test_provider_failure_finalizes_unscored_on_a_live_session(
    tmp_path: Path, episode_id: str
) -> None:
    """A provider error is not the environment's fault: no rescue, normal cleanup."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(error=RuntimeError("provider exhausted retries")),
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.rescue_required is False
    assert outcome.score is not None
    assert outcome.score.reason == "infrastructure_failure"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    errors = payloads(root, ErrorPayload)
    assert [error.category for error in errors] == ["provider"]
    # Cleanup ran on the live session rather than being minted as incomplete.
    assert "cleanup" in adapter.calls


@pytest.mark.asyncio
async def test_adapter_crash_poisons_runs_rescue_and_still_seals(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"execute": SupervisionError("worker died")}
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.rescue_required is True
    assert outcome.rescue_complete is True
    assert outcome.reportability_label == "cleanup_incomplete"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    errors = payloads(root, ErrorPayload)
    assert [error.category for error in errors] == ["adapter"]
    assert adapter.terminated


@pytest.mark.asyncio
async def test_crash_before_any_step_still_reaches_a_terminal(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"reset_start": SupervisionError("reset died")}
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "failed"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    assert outcome.score is not None and outcome.score.reason == "crash"


@pytest.mark.asyncio
async def test_scorer_failure_can_only_abandon_as_ambiguous(
    tmp_path: Path, episode_id: str
) -> None:
    """scoring_start is durable, so a scored intent cannot seal unscored."""

    adapter = FakeAdapter(tmp_path, episode_id, failures={"score": SupervisionError("scorer died")})
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "abandoned"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.abandonment is not None
    assert report.abandonment.reason == "ambiguous_finalization"
    # Rescue runs before abandonment: cloud safety must not wait on a writer.
    assert outcome.rescue_complete is True


@pytest.mark.asyncio
async def test_incomplete_cleanup_labels_the_run_and_requires_rescue(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id, cleanup_status="failed")
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    # The task scored, but a leaked resource still makes the episode failed.
    assert outcome.status == "failed"
    assert outcome.rescue_required is True
    assert outcome.reportability_label == "cleanup_incomplete"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    cleanups = payloads(root, CleanupPayload)
    assert [cleanup.rescue_required for cleanup in cleanups] == [True]


@pytest.mark.asyncio
async def test_cleanup_incomplete_outranks_unscored_in_the_label(
    tmp_path: Path, episode_id: str
) -> None:
    """Label precedence: cleanup_incomplete > budget_unreconciled > unscored."""

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        cleanup_status="failed",
        failures={"execute": SupervisionError("worker died")},
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    assert outcome.score is not None and outcome.score.status == "unscored"
    assert outcome.reportability_label == "cleanup_incomplete"


@pytest.mark.asyncio
async def test_unscored_score_artifact_is_never_a_zero_binary(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(error=RuntimeError("down")),
    )

    outcome = await runner.run()

    assert outcome.score is not None
    assert outcome.score.status == "unscored"
    assert outcome.score.binary is None


@pytest.mark.asyncio
async def test_failure_before_prepare_has_no_bundle_to_write(
    tmp_path: Path, episode_id: str
) -> None:
    """The manifest needs the cleanup plan prepare returns, so none exists yet."""

    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"prepare": SupervisionError("prepare died")}
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "failed_pre_bundle"
    assert outcome.bundle_root is None
    assert adapter.terminated


@pytest.mark.asyncio
async def test_rescue_descriptor_is_persisted_before_prepare_and_updated_after(
    tmp_path: Path, episode_id: str
) -> None:
    """Stage 1 unblocks prepare; stage 2 lands the real plan before reset_start."""

    from local_operator.evaluation.adapters.supervisor import load_pending_rescue

    seen: list[tuple[str, tuple[str, ...]]] = []
    adapter = FakeAdapter(tmp_path, episode_id)
    config = build_config(tmp_path)
    original = adapter._call_raw

    async def watching(method: Any, params: Any, result_type: Any, **kwargs: Any) -> Any:
        if method in ("prepare", "reset_start"):
            pending = load_pending_rescue(config.rescue_root)
            assert pending is not None, f"{method} ran with no persisted rescue"
            seen.append((method, tuple(a.action_id for a in pending.cleanup_plan.actions)))
        return await original(method, params, result_type, **kwargs)

    adapter._call_raw = watching  # type: ignore[method-assign]
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    # prepare sees the provisional close-session plan; reset_start, the side
    # effect boundary, already sees the adapter's real plan.
    assert seen == [
        ("prepare", ("close-session",)),
        ("reset_start", ("release",)),
    ]


@pytest.mark.asyncio
async def test_scored_zero_is_distinct_from_unscored(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id, score=ScoreArtifact(status="scored", binary=0))
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None
    assert outcome.score.status == "scored" and outcome.score.binary == 0
    assert outcome.reportability_label == "reportable"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid


@pytest.mark.asyncio
async def test_route_change_is_not_sealed_as_comparable(tmp_path: Path, episode_id: str) -> None:
    """A silent provider fallback is exactly what comparability exists to catch."""

    from local_operator.evaluation.evidence.models import RouteIdentity

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["finish"])
    model.route = RouteIdentity(
        provider_id="other-provider", route_id="other-route", model_id="other-model"
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.comparability_label == "route_changed"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid
    assert report.outcome is not None
    assert report.outcome.comparable is False


@pytest.mark.asyncio
async def test_pinned_route_stays_comparable(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.comparability_label == "comparable"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.outcome is not None
    assert report.outcome.comparable is True


# ---------------------------------------------------------------------------
# Managed context and guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_sees_each_prior_turn_with_its_batch(tmp_path: Path, episode_id: str) -> None:
    """The runner hands the client the turns it took, each closed with the
    batch that was executed on it, and the current turn still undecided."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step", "step", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    await runner.run()

    assert [len(history) for history in model.histories] == [1, 2, 3]
    third = model.histories[2]
    assert third[0].batch is not None and third[1].batch is not None
    assert third[2].batch is None
    assert third[0].observation.sequence == 0 and third[2].observation.sequence == 2
    # The current observation is the last turn's.
    assert third[2].observation is not None


@pytest.mark.asyncio
async def test_context_compaction_event_is_recorded_and_verifies(
    tmp_path: Path, episode_id: str
) -> None:
    from local_operator.evaluation.evidence.models import (
        ContextCompactionPayload,
        ModelRequestPayload,
    )

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step", "step", "finish"], compact_on=[1])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None and report.counters.compactions == 1
    kinds = _kinds(root)
    index = kinds.index("context_compaction")
    # Between the previous request's usage receipt and the next request.
    assert kinds[index - 1] == "observation"
    assert kinds[index - 2] == "environment_step"
    assert kinds[index + 1] == "model_request"
    assert kinds.count("context_compaction") == 1
    compaction = payloads(root, ContextCompactionPayload)[0]
    requests = payloads(root, ModelRequestPayload)
    assert compaction.previous_request_id == requests[0].request_id
    assert compaction.strategy == "context-full"
    assert compaction.frames_dropped == 2
    assert compaction.summary_artifact is not None
    assert (root / "artifacts" / compaction.summary_artifact.sha256).read_bytes() == (
        b"what happened so far"
    )
    assert all(request.prompt_cache_key == "lop-eval-test" for request in requests)


@pytest.mark.asyncio
async def test_guard_truncation_seals_scored_with_reason(tmp_path: Path, episode_id: str) -> None:
    """A floundering guard stops the episode the way ``max_steps`` does: the
    last step is truncated with the guard's code, the episode is SCORED on the
    state reached, and the bundle verifies."""

    from local_operator.evaluation.runner.guards import RepeatedBatchGuard

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=10, guards=(RepeatedBatchGuard(repeats=2),)),
        selector=selector(tmp_path),
        model=ScriptedModel(["type"] * 8),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    steps = payloads(root, EnvironmentStepPayload)
    assert len(steps) == 2
    assert [step.truncated for step in steps] == [False, True]
    assert [step.truncation_reason for step in steps] == [None, "repeated-batch"]


@pytest.mark.asyncio
async def test_a_waiting_model_runs_to_the_step_cap_under_default_guards(
    tmp_path: Path, episode_id: str
) -> None:
    """End to end for M1: a model that only waits (a slow load) reaches
    ``max_steps`` under the DEFAULT guards rather than being cut off as
    ``repeated-batch``; the truncation it gets is the honest step cap."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step"] * 12), max_steps=8
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert len(steps) == 8
    assert steps[-1].truncation_reason == "max-steps"


@pytest.mark.asyncio
async def test_max_steps_truncation_names_its_reason(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step"] * 6), max_steps=2
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert [step.truncation_reason for step in steps] == [None, "max-steps"]


@pytest.mark.asyncio
async def test_budget_cap_truncates_not_cancels(tmp_path: Path, episode_id: str) -> None:
    """A reached provider-cost cap is enforced as a scored truncation, where
    before it was only reported as an overrun after the fact."""

    # ScriptedModel bills 7 micro-USD per cycle: two cycles reach the cap.
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id, caps={"provider_usd_micros": 14}),
        build_config(tmp_path, max_steps=10),
        selector=selector(tmp_path),
        model=ScriptedModel(["step"] * 8),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert steps[-1].truncated is True
    assert steps[-1].truncation_reason == "budget-cap"
    assert len(steps) == 2


@pytest.mark.asyncio
async def test_ask_answer_rides_on_the_asking_turn(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["ask", "finish"])
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=model, responder=RecordingResponder("go left")
    )

    await runner.run()

    second = model.histories[1]
    assert second[0].ask_answer == "go left"
    assert second[0].batch is not None
    assert second[1].ask_answer is None


def test_episode_config_declares_no_frame_policy_it_cannot_enforce(tmp_path: Path) -> None:
    """The frame policy lives on the model client, which is built before the
    runner. A ``keep_recent_frames`` on the config was a knob a caller could
    set to 5 and get 3 (review round 1, m1); the config refuses it outright."""

    from dataclasses import fields

    from local_operator.evaluation.runner.episode import EpisodeConfig

    assert "keep_recent_frames" not in {field.name for field in fields(EpisodeConfig)}
    with pytest.raises(TypeError):
        build_config(tmp_path, keep_recent_frames=5)


@pytest.mark.asyncio
async def test_context_unrecoverable_seals_as_a_harness_error_not_a_provider_one(
    tmp_path: Path, episode_id: str
) -> None:
    """When the client cannot fit the context into the window even after
    shedding every stale observation, it refuses rather than send a request
    the provider will reject. The runner records that as a harness (adapter)
    error and seals UNSCORED -- honestly, with a valid bundle -- because a
    scored truncation is not representable: the last executed step's event was
    already written, and the verifier's one-step-per-batch rule forbids a
    corrective re-write (checked against the real verifier; see the PR thread).
    It must not be reclassified as a provider failure, which would blame an
    outage that did not happen."""

    from local_operator.evaluation.runner.provider_client import (
        ContextUnrecoverableError,
    )

    class RefusingModel:
        async def decide(self, observation: Any, history: Any) -> Any:
            raise ContextUnrecoverableError("context cannot fit the window")

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=RefusingModel(),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.score is not None and outcome.score.status == "unscored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    errors = payloads(root, ErrorPayload)
    assert len(errors) == 1
    assert errors[0].category == "adapter"
    assert errors[0].diagnostic_code == "contextunrecoverableerror"


# ---------------------------------------------------------------------------
# Rejected decisions: one bad reply is re-prompted, not fatal; the retry is
# in the evidence; the bound ends the episode as a MODEL failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_rejected_decision_is_re_prompted_and_the_corrected_batch_proceeds(
    tmp_path: Path, episode_id: str
) -> None:
    """Bundle ep-6ea01a117eee ended on its first decision (``frame_id '1'``
    against ``screen``) as a provider failure with $0 spent. A rejected reply
    is now recorded as a billed attempt with a retryable ``model`` error, the
    model is asked again for the SAME observation, and the corrected batch
    runs the episode to a scored, reportable seal."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "step", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.score is not None and outcome.score.status == "scored"
    assert outcome.reportability_label == "reportable"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    # Three billed calls, three triples: the rejected attempt is not hidden.
    assert report.counters is not None
    assert report.counters.model_request_count == 3
    assert report.counters.model_response_count == 3
    assert model.calls == 3
    # The re-prompt was for the same observation: identical history both times.
    assert model.histories[0] == model.histories[1]
    errors = payloads(root, ErrorPayload)
    assert [(e.category, e.diagnostic_code, e.retryable) for e in errors] == [
        ("model", "decision-rejected", True)
    ]
    assert errors[0].detail_artifact is not None
    # Journal order: the rejected attempt's triple, THEN its error, then the
    # corrected attempt's triple.
    kinds = _kinds(root)
    first_request = kinds.index("model_request")
    assert kinds[first_request : first_request + 7] == [
        "model_request",
        "model_response",
        "usage_cost",
        "error",
        "model_request",
        "model_response",
        "usage_cost",
    ]


@pytest.mark.asyncio
async def test_a_rejected_decision_publishes_the_reply_beside_the_diagnostic(
    tmp_path: Path, episode_id: str
) -> None:
    """A diagnostic alone cannot answer "what was the model attempting?".

    Bundle ep-ffda3fc88f81 recorded three ``decision-rejected`` errors whose
    artifacts held only the Pydantic complaint. Whether those were failed
    keyboard actions -- the difference between "the model never tried to type"
    and "the model tried and the schema refused it" -- was not answerable from
    the bundle, and answering it would have cost another paid run.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    errors = payloads(root, ErrorPayload)
    assert len(errors) == 1
    detail = errors[0].detail_artifact
    assert detail is not None
    recorded = (root / "artifacts" / detail.sha256).read_text()
    # Both halves, and labelled: a reader can tell the harness's refusal from
    # the model's own words.
    assert "unknown frame_id '1'" in recorded
    assert "--- rejected reply ---" in recorded
    assert '{"actions": [{"kind": "click", "frame_id": "1"}]}' in recorded


@pytest.mark.asyncio
async def test_a_rejection_without_a_captured_reply_records_the_diagnostic_alone(
    tmp_path: Path, episode_id: str
) -> None:
    """``reply`` is optional on the protocol boundary, so a client that does not
    capture it must degrade to the diagnostic rather than emit an empty section
    that reads as "the model said nothing"."""

    from local_operator.evaluation.runner.episode import _rejection_detail
    from local_operator.evaluation.runner.model import DecisionRejected

    assert _rejection_detail(DecisionRejected("refused")) == "refused"
    assert _rejection_detail(DecisionRejected("refused", reply="")) == "refused"
    assert "--- rejected reply ---" in _rejection_detail(DecisionRejected("refused", reply="{}"))


@pytest.mark.asyncio
async def test_exhausted_decision_retries_seal_as_a_model_failure(
    tmp_path: Path, episode_id: str
) -> None:
    """After ``max_decision_retries`` corrective re-prompts the model still
    has not produced a usable batch. That is the agent's failure, so the
    bundle says ``model`` / ``model_failure`` -- not ``provider`` (nothing was
    down) and not ``crash`` (nothing broke) -- and every billed attempt is in
    the evidence."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "reject", "reject", "step", "finish"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.rescue_required is False
    assert outcome.score is not None
    assert outcome.score.status == "unscored"
    assert outcome.score.reason == "model_failure"
    assert outcome.reportability_label == "unscored"
    assert outcome.diagnostic is not None and "3 attempt(s)" in outcome.diagnostic
    # Exactly the bound plus one: no fourth call.
    assert model.calls == 3
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None
    assert report.counters.model_request_count == 3
    errors = payloads(root, ErrorPayload)
    assert [(e.category, e.retryable) for e in errors] == [
        ("model", True),
        ("model", True),
        ("model", True),
        ("model", False),
    ]
    assert errors[-1].diagnostic_code == "modelfailure"
    # Cleanup ran on the live session: the environment was never at fault.
    assert "cleanup" in adapter.calls


@pytest.mark.asyncio
async def test_zero_decision_retries_restores_one_strike(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "finish"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=0),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.score is not None and outcome.score.reason == "model_failure"
    assert model.calls == 1


@pytest.mark.asyncio
async def test_crash_after_a_successful_step_still_seals(tmp_path: Path, episode_id: str) -> None:
    """A run interrupted mid-loop keeps the evidence it already bought.

    Regression for ep-ffda3fc88f81: a real paid episode drove 16 environment
    steps, took an unretryable adapter error on the 17th, and lost ALL of it --
    the bundle could neither seal nor be abandoned (``abandonment_failed``,
    score null) because the verifier required a stepped episode to end on a
    terminal step or a finish action, which is precisely what an interrupted
    episode cannot produce. The failure is not adapter-specific: the same shape
    killed a provider outage, so it is asserted here at the runner boundary.
    """

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError("worker died")},
        fail_after={"execute": 1},
    )
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "step", "finish"])
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    # The step that DID succeed is still in the bundle: that is the evidence
    # the old behaviour threw away.
    assert len(payloads(root, EnvironmentStepPayload)) == 1
    assert outcome.score is not None and outcome.score.reason == "crash"


@pytest.mark.asyncio
async def test_provider_failure_after_a_step_still_seals(tmp_path: Path, episode_id: str) -> None:
    """The same interruption through a non-adapter category."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step", "step", "finish"])
    original = model.decide
    calls = {"n": 0}

    async def decide(observation: Any, history: Any) -> Any:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("provider exhausted retries")
        return await original(observation, history)

    model.decide = decide  # type: ignore[method-assign]
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.status == "failed"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert outcome.score is not None and outcome.score.reason == "infrastructure_failure"


@pytest.mark.asyncio
async def test_fatal_error_records_a_bounded_diagnostic_detail(
    tmp_path: Path, episode_id: str
) -> None:
    """An unretryable failure must be diagnosable without a paid rerun.

    ``diagnostic_code`` is derived from the exception CLASS, so the real
    episode recorded only "rpcremoteerror" with a null ``detail_artifact`` --
    enough to bucket the failure, never enough to explain it.
    """

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError("worker died mid-observe")},
        fail_after={"execute": 1},
    )
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "step", "finish"])
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    fatal = [error for error in payloads(root, ErrorPayload) if not error.retryable]
    assert len(fatal) == 1
    assert fatal[0].diagnostic_code == "supervisionerror"
    detail = fatal[0].detail_artifact
    assert detail is not None
    text = (root / "artifacts" / detail.sha256).read_bytes().decode()
    assert text == "SupervisionError: worker died mid-observe"
    # Bounded: ``_diagnostic`` truncates, so a pathological message cannot
    # inflate the bundle.
    assert detail.byte_count <= 500


@pytest.mark.asyncio
async def test_fatal_diagnostic_detail_cannot_carry_a_resolved_secret(
    tmp_path: Path, episode_id: str
) -> None:
    """A secret in the failure message is withheld, and the bundle keeps a detail.

    The secret must never reach disk -- that assertion is unchanged and is the
    point of the test. What changed is the SHAPE of the refusal: the artifact
    used to be dropped entirely (``detail_artifact is None``), because the
    whole rendering was handed to ``publish_artifact`` and the canary scan
    rejected the write. A reader then got a fatal error with no detail at all,
    which is the exact opacity this PR exists to remove -- the secret was
    protected by destroying the diagnosis along with it.

    Now the value is withheld at the point it is rendered, so the artifact is
    published carrying the exception TYPE with the message replaced. The type
    cannot hold adapter or provider data and is what makes the failure
    bucketable, so keeping it costs nothing and is the whole difference between
    a diagnosable bundle and a blank one.
    """

    secret = "s3cret-value-that-must-never-land-in-a-bundle"
    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError(f"worker died using {secret}")},
        fail_after={"execute": 1},
    )
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=4),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "step", "finish"]),
        redactions=RedactionSet.from_resolved_values((secret,)),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    # The leak must not reach disk anywhere in the bundle.
    for path in root.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path
    # And refusing the detail must not cost the bundle its terminal.
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    fatal = [error for error in payloads(root, ErrorPayload) if not error.retryable]
    assert len(fatal) == 1
    # Published, not dropped: the reader still learns WHAT failed.
    reference = fatal[0].detail_artifact
    assert reference is not None
    published = (root / "artifacts" / reference.sha256).read_text()
    assert published == "SupervisionError: <withheld: matched a secret canary>"


@pytest.mark.asyncio
async def test_fatal_detail_withholds_a_secret_straddling_the_diagnostic_bound(
    tmp_path: Path, episode_id: str
) -> None:
    """A secret cut by ``_diagnostic``'s 500-char bound must still be withheld.

    Round 1 fixed this ordering in the WORKER; the parent had the identical
    inversion. ``_diagnostic`` truncated to 500 characters and only then were
    the bytes handed to ``publish_artifact``, whose scan is a SUBSTRING check --
    so a severed canary stops matching and the surviving prefix is sealed into
    the bundle. Measured before the fix: 20 characters of a canonical AWS key,
    and 63 of an API token echoed in a provider error URL.

    The sibling test above uses a SHORT secret that fits inside the bound
    entirely, which is exactly why it passed while this leaked: the position of
    the secret relative to the cut is the whole property under test, so the
    padding here is load-bearing rather than decorative.
    """

    secret = "AKIAIOSFODNN7EXAMPLE" + "QWERTYUIOPASDFGH1234"
    # Pad so the 500-character cut lands INSIDE the credential. The rendering
    # is ``"SupervisionError: " + message`` (18 chars) plus this prefix (16),
    # so 446 filler characters leave exactly 20 characters of the key inside
    # the bound -- the arithmetic is the test, and an off-by-a-few pushes the
    # whole secret past the cut where nothing leaks even when the ordering is
    # wrong.
    message = "adapter failed: " + ("x" * 446) + secret + " (will not retry)"
    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError(message)},
        fail_after={"execute": 1},
    )
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=4),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "step", "finish"]),
        redactions=RedactionSet.from_resolved_values((secret,)),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    # No PREFIX of the credential reaches disk either -- a whole-value check
    # passes against the very truncation this test exists to catch.
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        for length in range(8, len(secret) + 1):
            assert secret[:length].encode() not in blob, (path, length)
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]


@pytest.mark.asyncio
async def test_provider_failure_withholds_a_secret_straddling_the_bound(
    tmp_path: Path, episode_id: str
) -> None:
    """A PROVIDER error's text is scanned before it is cut, not after.

    Round 2 (R2-1) found the third instance of the same inversion: the provider
    path rendered ``_diagnostic(error)`` with no redaction set on the way into
    ``_ProviderFailure``, and ``_finalize_failure`` later published that already
    truncated message to the sealed bundle. The cut severs the canary, so the
    scan at the publish site -- and ``publish_artifact``'s own scan -- both
    return clean on the surviving prefix.

    This is the case the runner never sees redacted from anywhere else: a
    provider exception is raised on the way to the model, so the adapter worker
    (whose own redaction was fixed in round 1) never touches it.

    The placement is SWEPT rather than fixed, and that is the point. A single
    offset proves nothing here: the message crosses two independent 500-char
    cuts (this site, then ``_failure_detail``), so the surviving fragment is a
    non-monotonic function of where the secret sits. Measured against the bug,
    an offset of 20 leaks only 2 characters -- under any sane prefix threshold
    that reads as CLEAN -- while an offset of 40 leaks 26. Two prior audits and
    the first draft of this test were all defeated by exactly that: a negative
    result at one offset was mistaken for a negative result. The sweep is the
    regression guard; pinning one number would re-arm the trap.
    """

    secret = "AKIAIOSFODNN7EXAMPLE" + "QWERTYUIOPASDFGH1234"
    # 39, and the exact value is evidence rather than taste. The message
    # crosses TWO independent 500-char cuts (this site, then
    # ``_failure_detail``), so the fragment that survives both is non-monotonic
    # in the offset: measured against the bug, offsets 22-39 leak 8-25
    # characters into the sealed artifact, while 40 leaks NOTHING because the
    # downstream scan happens to see enough of the value to match. Picking 40
    # produced a test that passed against the bug it was written for.
    survivors = 39
    # The rendering is ``"RuntimeError: " + message`` (14 characters).
    padding = 500 - 14 - survivors
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=4),
        selector=selector(tmp_path),
        model=ScriptedModel(error=RuntimeError("x" * padding + secret + " tail")),
        redactions=RedactionSet.from_resolved_values((secret,)),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert [error.category for error in payloads(root, ErrorPayload)] == ["provider"]
    # No PREFIX reaches disk. Asserting the whole value would pass against the
    # very truncation this test exists to catch.
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        for length in range(8, len(secret) + 1):
            assert secret[:length].encode() not in blob, (path, length)


@pytest.mark.asyncio
async def test_detail_publish_os_error_cannot_escape_the_failure_handler(
    tmp_path: Path,
    episode_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk failure while publishing the fatal-error detail must not lose the terminal.

    Regression for the round-1 review (R1-1): ``publish_artifact`` poisons the
    writer and re-raises a RAW ``OSError`` on ambiguous I/O (ENOSPC/EIO), and
    ``_publish`` converts only ``EvidenceError`` -- so before the catch at the
    detail site was widened, an OSError there escaped ``_finalize_failure`` and
    propagated out of ``run()``: the caller got a bare exception, no
    ``EpisodeOutcome``, and a bundle with its steps but no terminal. That is the
    exact "failure handler destroys the bundle's terminal" shape this PR exists
    to remove, gated on a disk error instead of an adapter error.

    The promise is best-effort publication, never best-effort terminal: the
    detail may be lost, but ``run()`` must still return a structured outcome and
    the bundle must stay recoverable (a fresh process can still abandon it).
    """

    detail_bytes = b"SupervisionError: worker died"
    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError("worker died")},
        fail_after={"execute": 1},
    )
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "step", "finish"])
    )

    real_publish = EvidenceWriter.publish_artifact

    def faulty_publish(self: Any, source: Any, *, media_type: str, **kwargs: Any) -> Any:
        if bytes(source) == detail_bytes:
            # The store's real contract on ambiguous I/O: poison, then re-raise
            # the raw OSError (store.py publish_artifact).
            self._poison()  # pyright: ignore[reportPrivateUsage]
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_publish(self, source, media_type=media_type, **kwargs)

    monkeypatch.setattr(EvidenceWriter, "publish_artifact", faulty_publish)

    # Pre-fix this raised OSError out of run(); the fix returns an outcome.
    outcome = await runner.run()

    assert outcome.status == "abandonment_failed"
    root = outcome.bundle_root
    assert root is not None
    # The evidence the episode already bought is still on disk.
    report = verify_bundle(root)
    assert len([e for e in report.events if isinstance(e.payload, EnvironmentStepPayload)]) == 1
    # And the bundle is recoverable: a fresh process can still abandon it, so
    # the terminal is reachable even though this run could not record it.
    recovered = EvidenceWriter.open_for_abandon(root, RedactionSet.from_resolved_values(()))
    try:
        record = recovered.abandon("infrastructure_failure", "evidence-write-failed")
    finally:
        recovered.close()
    assert record.reason == "infrastructure_failure"
    assert verify_bundle(root).terminal_state == "abandoned"


# ---------------------------------------------------------------------------
# Disclosed infra values must not be silently dropped by an older adapter
# ---------------------------------------------------------------------------
#
# The defect these cover is a FALSE disclosure, which is strictly worse than a
# missing one: the caller stamps "this run used m5.xlarge" into the sealed
# manifest from the CLI string alone, while an adapter build that predates the
# knob resolves t3.xlarge and drops it. The bundle then verifies, the run exits
# 0, and a reader reports a burstable-starved score as starvation-corrected.
# ``inspect_requirements`` is the only signal that separates two builds sharing
# one version string, so the runner binds it and fails closed.


def _spec_with_infra(episode_id: str, *names: str) -> Any:
    spec = build_spec(episode_id)
    object.__setattr__(
        spec,
        "infra_values",
        tuple(
            ScopedInfraValue(name=name, purpose="benchmark_compute", value="m5.xlarge")
            for name in names
        ),
    )
    return spec


def _infra_runner(
    tmp_path: Path, episode_id: str, spec: Any, adapter: FakeAdapter
) -> EpisodeRunner:
    return EpisodeRunner(
        spec,
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )


@pytest.mark.asyncio
async def test_a_disclosed_infra_value_an_adapter_does_not_declare_is_refused(
    tmp_path: Path, episode_id: str
) -> None:
    """The reviewer's reproduction, as a test: old adapter + new host script.

    A ``FakeAdapter`` declaring no requirements is exactly how a build that
    predates ``AWS_INSTANCE_TYPE`` answers for that name. Pre-fix this ran to
    a sealed, valid bundle claiming hardware the episode never used; the
    refusal must land BEFORE ``prepare``, so nothing is allocated and there is
    no bundle to mislead a reader.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, "AWS_INSTANCE_TYPE"), adapter
    ).run()

    assert outcome.status == "failed_pre_bundle"
    assert outcome.bundle_root is None
    assert "AWS_INSTANCE_TYPE" in (outcome.diagnostic or "")
    assert "UndeclaredDisclosedInfra" in (outcome.diagnostic or "")
    # The side-effect boundary was never crossed and the worker was reaped.
    assert "prepare" not in adapter.calls and "reset_start" not in adapter.calls
    assert adapter.terminated


@pytest.mark.asyncio
async def test_a_declared_disclosed_infra_value_runs_normally(
    tmp_path: Path, episode_id: str
) -> None:
    """The inverse direction: a rebuilt adapter DOES declare it, so it runs.

    Without this the gate could pass by refusing everything, which would break
    the very workflow the knob exists for.
    """

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        declared_requirements=(
            Requirement(
                requirement_id="AWS_INSTANCE_TYPE",
                kind="infra",
                name="AWS_INSTANCE_TYPE",
                required=False,
            ),
        ),
    )
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, "AWS_INSTANCE_TYPE"), adapter
    ).run()

    assert outcome.status == "completed", outcome.diagnostic
    assert "prepare" in adapter.calls and "reset_start" in adapter.calls


@pytest.mark.asyncio
async def test_an_undeclared_ordinary_infra_value_is_not_refused(
    tmp_path: Path, episode_id: str
) -> None:
    """The gate is a narrow allowlist, and that scoping is load-bearing.

    ``inspect_requirements`` runs BEFORE the task is named, so it returns only
    the adapter's unconditional baseline -- a task-conditional requirement
    (OSWORLD_PROXY_ENDPOINT for a proxy task) is legitimately absent from it.
    A blanket "refuse every undeclared infra value" rule would therefore break
    correct proxy runs, which is why only values whose silent drop corrupts the
    evidence record are gated.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, "OSWORLD_PROXY_ENDPOINT"), adapter
    ).run()

    assert outcome.status == "completed", outcome.diagnostic


@pytest.mark.asyncio
async def test_a_root_volume_override_an_adapter_does_not_declare_is_refused(
    tmp_path: Path, episode_id: str
) -> None:
    """The second disclosed value inherits the gate, not just the allowlist.

    Adding a value to the disclosure mechanism is exactly the drift a reviewer
    predicted when the stamp and the allowlist were two hardcoded lists: a
    value stamped into the manifest but ungated is a FALSE disclosure sealed
    in a bundle that verifies. The refusal must land before ``prepare``, so
    nothing is allocated and no bundle exists to mislead a reader.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, "AWS_ROOT_VOLUME_SIZE"), adapter
    ).run()

    assert outcome.status == "failed_pre_bundle"
    assert outcome.bundle_root is None
    assert "AWS_ROOT_VOLUME_SIZE" in (outcome.diagnostic or "")
    assert "UndeclaredDisclosedInfra" in (outcome.diagnostic or "")
    assert "prepare" not in adapter.calls and "reset_start" not in adapter.calls
    assert adapter.terminated


@pytest.mark.asyncio
async def test_a_declared_root_volume_override_runs_normally(
    tmp_path: Path, episode_id: str
) -> None:
    """The inverse: a rebuilt adapter declares it, so the knob is usable.

    Without this the gate could pass by refusing everything, which would break
    the very workflow the override exists for.
    """

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        declared_requirements=(
            Requirement(
                requirement_id="AWS_ROOT_VOLUME_SIZE",
                kind="infra",
                name="AWS_ROOT_VOLUME_SIZE",
                required=False,
            ),
        ),
    )
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, "AWS_ROOT_VOLUME_SIZE"), adapter
    ).run()

    assert outcome.status == "completed", outcome.diagnostic
    assert "prepare" in adapter.calls and "reset_start" in adapter.calls


def test_every_disclosed_infra_value_is_gated_and_stamped_from_one_table() -> None:
    """The gate and the stamp must not drift apart, and this is what binds them.

    ``scripts/run_episode.py`` used to repeat the disclosed names in its own
    hardcoded branch. Nothing tied the two lists together, so a third value
    could be stamped without a gate (a disclosure nothing verifies) or gated
    without a stamp (a check nothing records) -- neither visible by reading
    either file alone. Both now derive from ``DISCLOSED_INFRA_METADATA_KEYS``;
    this asserts the derivation rather than the current contents, so it keeps
    holding when a fourth value is added.
    """

    import scripts.run_episode as run_episode
    from local_operator.evaluation.runner.episode import _DISCLOSED_INFRA_VALUES

    # The gate's allowlist is exactly the table's keys -- not a copy of them.
    assert _DISCLOSED_INFRA_VALUES == frozenset(DISCLOSED_INFRA_METADATA_KEYS)

    # And every gated name reaches the manifest under its declared key, which
    # is the half a reviewer cannot see from episode.py alone.
    for name, key in DISCLOSED_INFRA_METADATA_KEYS.items():
        metadata = run_episode._infra_disclosure_metadata([f"{name}=some-value"])
        assert metadata == {key: "some-value"}, name

    # The keys are distinct: two values sharing one key would have the second
    # silently overwrite the first's disclosure.
    keys = list(DISCLOSED_INFRA_METADATA_KEYS.values())
    assert len(keys) == len(set(keys))


def test_an_undisclosed_infra_value_is_never_stamped() -> None:
    """The stamp is an allowlist too: an ordinary infra value must not leak
    into the manifest, and a value with no ``=`` must not be stamped empty."""

    import scripts.run_episode as run_episode

    assert run_episode._infra_disclosure_metadata(["OSWORLD_TTL_SECONDS=900"]) == {}
    assert run_episode._infra_disclosure_metadata(["AWS_ROOT_VOLUME_SIZE"]) == {}
    assert run_episode._infra_disclosure_metadata([]) == {}
    # Both at once, each under its own key.
    assert run_episode._infra_disclosure_metadata(
        ["AWS_INSTANCE_TYPE=m5.xlarge", "AWS_ROOT_VOLUME_SIZE=120", "OSWORLD_TTL_SECONDS=900"]
    ) == {"aws_instance_type_override": "m5.xlarge", "aws_root_volume_size_override": "120"}

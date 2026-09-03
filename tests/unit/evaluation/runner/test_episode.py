"""Event mapping and terminal coherence for one episode.

Every test here drives the REAL ``VerifiedAdapterSession``, the REAL lifecycle
authorities and a REAL ``EvidenceWriter``; only the subprocess boundary is
faked. The standing assertion is that each terminal path leaves a bundle the
independent verifier accepts, because a runner that writes evidence no verifier
will take is indistinguishable from one that writes none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.supervisor import SupervisionError
from local_operator.evaluation.evidence.models import (
    CleanupPayload,
    EnvironmentStepPayload,
    ErrorPayload,
    ObservationPayload,
    ScoreArtifact,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.runner.episode import EpisodeRunner
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

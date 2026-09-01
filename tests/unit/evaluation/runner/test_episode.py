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
from local_operator.evaluation.evidence.models import ScoreArtifact
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.runner.episode import EpisodeRunner
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    RecordingResponder,
    ScriptedModel,
    build_config,
    build_spec,
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
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"])
    )

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
async def test_step_event_precedes_its_output_observation(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"])
    )

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
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"])
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    observations = [
        event.payload.sequence
        for event in verify_bundle(root).events
        if event.kind == "observation"
    ]
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
    steps = [
        event.payload
        for event in verify_bundle(root).events
        if event.kind == "environment_step"
    ]
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
    errors = [
        event.payload for event in verify_bundle(root).events if event.kind == "error"
    ]
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
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"])
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.rescue_required is True
    assert outcome.rescue_complete is True
    assert outcome.reportability_label == "cleanup_incomplete"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    errors = [event.payload for event in report.events if event.kind == "error"]
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

    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"score": SupervisionError("scorer died")}
    )
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
    cleanups = [
        event.payload for event in verify_bundle(root).events if event.kind == "cleanup"
    ]
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
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"])
    )

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
            seen.append(
                (method, tuple(a.action_id for a in pending.cleanup_plan.actions))
            )
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
async def test_scored_zero_is_distinct_from_unscored(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(
        tmp_path, episode_id, score=ScoreArtifact(status="scored", binary=0)
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None
    assert outcome.score.status == "scored" and outcome.score.binary == 0
    assert outcome.reportability_label == "reportable"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid

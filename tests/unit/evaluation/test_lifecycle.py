"""Transition and cleanup contracts for benchmark-neutral evaluation episodes."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from local_operator.evaluation.lifecycle import (
    CleanupAction,
    CleanupPlan,
    CleanupResult,
    EpisodeLifecycle,
    ScoreReceipt,
    SideEffectPermit,
    aggregate_cleanup,
    record_cleanup,
)
from local_operator.evaluation.protocol import ArtifactRef
from local_operator.evaluation.receipts import (
    BUDGET_RESOURCES,
    AvailableUsage,
    BudgetAuthorization,
    BudgetCommitment,
    BudgetReconciliation,
    BudgetReservation,
    CappedAllowance,
    ComputeRequirement,
    DependencyPlan,
    RedactionSet,
    ResourceAmount,
    SealedPreflight,
    commit_budget,
    reconcile_budget,
    record_preflight,
    reserve_budget,
    seal_preflight,
)

REPO = Path(__file__).resolve().parents[3]
DIGEST = "0123456789abcdef" * 4


def _plan() -> DependencyPlan:
    return DependencyPlan(
        release_id="release-1",
        task_id="task-1",
        attempt_id="attempt-1",
        requirements=(
            ComputeRequirement(
                requirement_id="compute",
                necessity="required",
                reportability="required",
                cpu_class="standard",
                memory_class="standard",
                disk_bytes=1_000,
            ),
        ),
    )


def _seal(plan: DependencyPlan) -> SealedPreflight:
    receipt = record_preflight(
        plan,
        "compute",
        status="pass",
        evidence={"probe": "safe"},
        duration_ms=1,
    )
    return seal_preflight(plan, (receipt,), RedactionSet.from_resolved_values(()))


def _budget() -> BudgetAuthorization:
    return BudgetAuthorization(
        episode_id="episode-1",
        allowances=tuple(
            CappedAllowance(resource=resource, value=100, reporting="required")
            for resource in BUDGET_RESOURCES
        ),
    )


def _cleanup_plan() -> CleanupPlan:
    return CleanupPlan(
        episode_id="episode-1",
        actions=(
            CleanupAction(
                action_id="session",
                kind="close_session",
                resource_ref="session-lease",
                timeout_ms=1_000,
                max_attempts=3,
            ),
            CleanupAction(
                action_id="volume",
                kind="delete_volume",
                resource_ref="volume-lease",
                timeout_ms=5_000,
                max_attempts=2,
            ),
        ),
    )


def _cleanup(
    plan: CleanupPlan,
    *,
    session: Literal["not_needed", "attempted", "succeeded", "failed"] = "succeeded",
    volume: Literal["not_needed", "attempted", "succeeded", "failed"] = "not_needed",
) -> CleanupResult:
    receipts = (
        record_cleanup(
            plan,
            "session",
            status=session,
            evidence_code="adapter-confirmed",
            duration_ms=2,
        ),
        record_cleanup(
            plan,
            "volume",
            status=volume,
            evidence_code="adapter-confirmed",
            duration_ms=1,
        ),
    )
    return aggregate_cleanup(plan, receipts)


def _reservation(budget: BudgetAuthorization):  # type annotation inferred from factory
    return reserve_budget(
        budget,
        "episode-reservation",
        tuple(ResourceAmount(resource=resource, value=10) for resource in BUDGET_RESOURCES),
    )


def _commitment(budget: BudgetAuthorization, reservation: Any) -> BudgetCommitment:
    return commit_budget(budget, (reservation,))


def _reconciliation(
    budget: BudgetAuthorization,
    reservation: Any,
    *,
    unavailable: bool = False,
) -> BudgetReconciliation:
    usage = [AvailableUsage(resource=resource, value=5) for resource in BUDGET_RESOURCES]
    if unavailable:
        from local_operator.evaluation.receipts import UnavailableUsage

        return reconcile_budget(
            budget,
            (reservation,),
            (
                UnavailableUsage(resource="provider_input_tokens", reason="provider omitted usage"),
                *usage[1:],
            ),
        )
    return reconcile_budget(budget, (reservation,), usage)


def _score(plan: DependencyPlan) -> ScoreReceipt:
    return ScoreReceipt(
        episode_id="episode-1",
        plan_id=plan.plan_id,
        score_artifact=ArtifactRef(
            sha256=DIGEST,
            media_type="application/json",
            byte_count=100,
        ),
        finalized_at_ms=10,
    )


def _authorized() -> tuple[
    DependencyPlan,
    BudgetAuthorization,
    CleanupPlan,
    EpisodeLifecycle,
    SideEffectPermit,
]:
    plan = _plan()
    seal = _seal(plan)
    budget = _budget()
    cleanup = _cleanup_plan()
    episode = EpisodeLifecycle.planned(
        episode_id="episode-1",
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    ).preflight(seal)
    authorized, permit = episode.authorize(seal, budget)
    return plan, budget, cleanup, authorized, permit


def test_cleanup_plan_is_symbolic_bounded_canonical_and_stable() -> None:
    plan = _cleanup_plan()
    shuffled = CleanupPlan(episode_id="episode-1", actions=tuple(reversed(plan.actions)))
    assert shuffled.cleanup_plan_id == plan.cleanup_plan_id
    assert CleanupPlan.from_canonical_json(plan.to_canonical_json()) == plan
    assert b"command" not in plan.to_canonical_json()
    assert b"api_params" not in plan.to_canonical_json()
    with pytest.raises(ValidationError, match="duplicate action IDs"):
        CleanupPlan(episode_id="episode-1", actions=(plan.actions[0], plan.actions[0]))
    with pytest.raises(ValidationError):
        CleanupAction.model_validate(
            {
                **plan.actions[0].model_dump(),
                "raw_command": "rm -rf /",
            }
        )


def test_cleanup_requires_exactly_one_receipt_and_attempted_is_not_clean() -> None:
    plan = _cleanup_plan()
    receipt = record_cleanup(
        plan,
        "session",
        status="succeeded",
        evidence_code="confirmed",
        duration_ms=1,
    )
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_cleanup(plan, (receipt,))
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_cleanup(plan, (receipt, receipt))
    attempted = _cleanup(plan, session="attempted")
    assert attempted.rescue_required
    assert attempted.incomplete_action_ids == ("session",)
    with pytest.raises(ValidationError, match="validated receipts"):
        CleanupResult.from_canonical_json(attempted.to_canonical_json())
    clean = _cleanup(plan)
    assert not clean.rescue_required


def test_permit_cannot_be_constructed_or_deserialized_by_callers() -> None:
    plan, budget, _cleanup_plan_value, _episode, permit = _authorized()
    payload = permit.model_dump()
    with pytest.raises(ValidationError, match="only be minted"):
        SideEffectPermit.model_validate(payload)
    with pytest.raises(ValidationError, match="only be minted"):
        SideEffectPermit.from_canonical_json(permit.to_canonical_json())
    with pytest.raises(ValidationError):
        SideEffectPermit(
            episode_id="episode-1",
            plan_id=plan.plan_id,
            preflight_seal_id=DIGEST,
            budget_id=budget.budget_id,
            permit_id=DIGEST,
        )


def test_side_effect_start_is_impossible_before_seal_permit_and_reservation() -> None:
    plan = _plan()
    budget = _budget()
    cleanup = _cleanup_plan()
    planned = EpisodeLifecycle.planned(
        episode_id="episode-1",
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    )
    _, _, _, authorized, permit = _authorized()
    reservation = _reservation(budget)
    with pytest.raises(ValueError, match="illegal episode transition"):
        planned.start(permit, budget, _commitment(budget, reservation))
    with pytest.raises(ValueError, match="factory authority"):
        authorized.start(permit, budget, BudgetCommitment.model_construct())
    forged_payload = {**permit.model_dump(), "episode_id": "episode-other"}
    with pytest.raises(ValidationError):
        SideEffectPermit.model_validate(forged_payload)
    running = authorized.start(permit, budget, _commitment(budget, reservation))
    assert running.state == "running"
    assert running.reservation_ids == (reservation.reservation_id,)


def test_legal_happy_path_requires_cost_score_and_clean_result() -> None:
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    running = authorized.start(permit, budget, _commitment(budget, reservation))
    finalizing = running.begin_finalization()
    reconciliation = _reconciliation(budget, reservation)
    cleaning = finalizing.finish_finalization(reconciliation, _score(plan))
    assert cleaning.state == "cleaning"
    completed = cleaning.finish_cleanup(_cleanup(cleanup))
    assert completed.state == "completed"
    assert completed.reconciliation_id == reconciliation.reconciliation_id
    assert completed.score_id is not None
    assert completed.rescue_required is False
    with pytest.raises(ValidationError, match="factory minted"):
        EpisodeLifecycle.from_canonical_json(completed.to_canonical_json())


def test_unreportable_usage_or_incomplete_cleanup_cannot_complete() -> None:
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    finalizing = authorized.start(
        permit, budget, _commitment(budget, reservation)
    ).begin_finalization()
    unreportable = _reconciliation(budget, reservation, unavailable=True)
    cleaning = finalizing.finish_finalization(unreportable, _score(plan))
    failed = cleaning.finish_cleanup(_cleanup(cleanup))
    assert failed.state == "failed"
    assert failed.failure_kind == "unreportable"
    reportable = _reconciliation(budget, reservation)
    cleaning = finalizing.finish_finalization(reportable, _score(plan))
    rescue = cleaning.finish_cleanup(_cleanup(cleanup, volume="failed"))
    assert rescue.state == "failed"
    assert rescue.failure_kind == "cleanup"
    assert rescue.rescue_required


def test_crash_and_cancel_after_running_must_flow_through_cleanup() -> None:
    _plan_value, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    running = authorized.start(permit, budget, _commitment(budget, reservation))
    crashed = running.crash("adapter process exited")
    assert crashed.state == "cleaning"
    failed = crashed.finish_cleanup(_cleanup(cleanup))
    assert failed.state == "failed"
    assert failed.cleanup_result_id is not None
    cancelled = running.cancel("operator requested cancellation")
    assert cancelled.state == "cleaning"
    terminal = cancelled.finish_cleanup(_cleanup(cleanup))
    assert terminal.state == "cancelled"
    assert terminal.cleanup_result_id is not None


def test_preflight_and_infrastructure_failures_are_unscored() -> None:
    plan = _plan()
    budget = _budget()
    cleanup = _cleanup_plan()
    planned = EpisodeLifecycle.planned(
        episode_id="episode-1",
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    )
    failed = planned.fail_before_running(kind="preflight", reason="display unavailable")
    assert failed.state == "failed"
    assert failed.score_id is None
    with pytest.raises(ValueError, match="illegal"):
        failed.begin_finalization()
    preflighted = planned.preflight(_seal(plan))
    infrastructure = preflighted.fail_before_running(
        kind="infrastructure", reason="allocator unavailable"
    )
    assert infrastructure.score_id is None


def test_ambiguous_finalization_is_terminal_after_cleanup_and_cannot_rescore() -> None:
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    finalizing = authorized.start(
        permit, budget, _commitment(budget, reservation)
    ).begin_finalization()
    ambiguous = finalizing.mark_ambiguous_finalization(
        _reconciliation(budget, reservation),
        "judge response committed but acknowledgement was lost",
    )
    assert ambiguous.state == "cleaning"
    with pytest.raises(ValueError, match="illegal"):
        ambiguous.finish_finalization(_reconciliation(budget, reservation), _score(plan))
    failed = ambiguous.finish_cleanup(_cleanup(cleanup))
    assert failed.state == "failed"
    assert failed.failure_kind == "ambiguous_finalization"


@pytest.mark.parametrize(
    ("state", "operation"),
    [
        ("planned", "begin_finalization"),
        ("preflighted", "begin_finalization"),
        ("authorized", "begin_finalization"),
        ("running", "preflight"),
        ("finalizing", "begin_finalization"),
        ("cleaning", "begin_finalization"),
        ("completed", "begin_finalization"),
        ("failed", "begin_finalization"),
        ("cancelled", "begin_finalization"),
    ],
)
def test_illegal_transition_table(state: str, operation: str) -> None:
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    running = authorized.start(permit, budget, _commitment(budget, reservation))
    reconciliation = _reconciliation(budget, reservation)
    finalizing = running.begin_finalization()
    cleaning = finalizing.finish_finalization(reconciliation, _score(plan))
    episodes = {
        "planned": EpisodeLifecycle.planned(
            episode_id="episode-1",
            plan_id=plan.plan_id,
            budget_id=budget.budget_id,
            cleanup_plan_id=cleanup.cleanup_plan_id,
        ),
        "preflighted": EpisodeLifecycle.planned(
            episode_id="episode-1",
            plan_id=plan.plan_id,
            budget_id=budget.budget_id,
            cleanup_plan_id=cleanup.cleanup_plan_id,
        ).preflight(_seal(plan)),
        "authorized": authorized,
        "running": running,
        "finalizing": finalizing,
        "cleaning": cleaning,
        "completed": cleaning.finish_cleanup(_cleanup(cleanup)),
        "failed": running.crash("crash").finish_cleanup(_cleanup(cleanup)),
        "cancelled": running.cancel("cancel").finish_cleanup(_cleanup(cleanup)),
    }
    episode = episodes[state]
    if operation == "preflight":
        with pytest.raises(ValueError, match="illegal"):
            episode.preflight(_seal(plan))
    else:
        with pytest.raises(ValueError, match="illegal|lacks transition authority"):
            episode.begin_finalization()


def test_direct_snapshots_cannot_claim_completion_without_all_evidence() -> None:
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    base = authorized.start(permit, budget, _commitment(budget, reservation)).model_dump()
    for missing in ("preflight_seal_id", "permit_id", "reconciliation_id", "score_id"):
        payload = {
            **base,
            "state": "completed",
            "terminal_intent": "complete",
            "reconciliation_id": DIGEST,
            "reconciliation_reportable": True,
            "score_id": DIGEST,
            "cleanup_result_id": DIGEST,
            "rescue_required": False,
            missing: None,
        }
        with pytest.raises(ValidationError):
            EpisodeLifecycle.model_validate(payload)
    assert cleanup.cleanup_plan_id == base["cleanup_plan_id"]


def test_imports_are_isolated_and_evaluation_root_remains_inert() -> None:
    forbidden = (
        "PIL",
        "boto3",
        "botocore",
        "gymnasium",
        "osworld",
        "OSWorld",
        "subprocess",
        "local_operator.config",
        "local_operator.providers",
        "local_operator.tools",
        "local_operator.tui",
        "local_operator.mobile",
    )
    for module in (
        "local_operator.evaluation.receipts",
        "local_operator.evaluation.lifecycle",
    ):
        imported = _fresh_import_modules(module)
        assert not {
            name
            for name in imported
            if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
        }
    root_imports = _fresh_import_modules("local_operator.evaluation")
    assert "local_operator.evaluation.receipts" not in root_imports
    assert "local_operator.evaluation.lifecycle" not in root_imports
    for startup in ("local_operator.cli", "local_operator.session_factory"):
        imported = _fresh_import_modules(startup)
        assert "local_operator.evaluation.receipts" not in imported
        assert "local_operator.evaluation.lifecycle" not in imported


def _fresh_import_modules(module: str) -> set[str]:
    probe = (
        "import importlib,json,sys;"
        "importlib.import_module(sys.argv[1]);"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, module],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3_000:]
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


def test_evaluation_root_source_stays_inert() -> None:
    root = importlib.import_module("local_operator.evaluation")
    assert not hasattr(root, "EpisodeLifecycle")
    assert not hasattr(root, "DependencyPlan")


def test_cleanup_receipt_replay_rejects_changed_plan_or_action() -> None:
    plan = _cleanup_plan()
    receipt = record_cleanup(
        plan,
        "session",
        status="succeeded",
        evidence_code="confirmed",
        duration_ms=1,
    )
    changed_action = CleanupAction.model_validate(
        {
            **plan.actions[0].model_dump(exclude={"action_digest"}),
            "resource_ref": "other-lease",
        }
    )
    changed_plan = CleanupPlan(
        episode_id=plan.episode_id,
        actions=(changed_action, plan.actions[1]),
    )
    other = record_cleanup(
        changed_plan,
        "volume",
        status="not_needed",
        evidence_code="confirmed",
        duration_ms=1,
    )
    with pytest.raises(ValueError, match="another cleanup plan|another action"):
        aggregate_cleanup(changed_plan, (receipt, other))
    with pytest.raises(ValidationError):
        CleanupAction(
            action_id="zero",
            kind="close_session",
            resource_ref="lease",
            timeout_ms=0,
            max_attempts=1,
        )


def test_lifecycle_authority_rejects_every_public_construction_and_mutation_path() -> None:
    import copy
    import pickle

    plan, budget, cleanup, authorized, _permit = _authorized()
    payload = authorized.model_dump()
    with pytest.raises(ValidationError, match="factory minted"):
        EpisodeLifecycle.model_validate(payload)
    with pytest.raises(ValidationError, match="factory minted"):
        EpisodeLifecycle.model_validate_json(authorized.to_canonical_json())
    with pytest.raises(ValidationError, match="factory minted"):
        EpisodeLifecycle.from_canonical_json(authorized.to_canonical_json())
    forged = EpisodeLifecycle.model_construct(**payload)
    with pytest.raises(ValueError, match="lacks transition authority"):
        forged.begin_finalization()
    with pytest.raises(TypeError, match="cannot be updated"):
        authorized.model_copy(update={"state": "completed"})
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(authorized)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(authorized)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(authorized)
    assert authorized.previous_state_id is not None
    assert authorized.state_id != authorized.previous_state_id
    assert plan.plan_id == payload["plan_id"]
    assert budget.budget_id == payload["budget_id"]
    assert cleanup.cleanup_plan_id == payload["cleanup_plan_id"]


def test_direct_overcap_reservation_and_forged_commitment_cannot_start() -> None:
    _plan_value, budget, _cleanup_value, authorized, permit = _authorized()
    overcap = BudgetReservation(
        episode_id=budget.episode_id,
        budget_id=budget.budget_id,
        reservation_key="forged-overcap",
        amounts=(ResourceAmount(resource="guest_actions", value=101),),
    )
    with pytest.raises(ValueError, match="exceeds"):
        commit_budget(budget, (overcap,))
    legitimate = _reservation(budget)
    commitment = _commitment(budget, legitimate)
    forged = BudgetCommitment.model_construct(**commitment.model_dump())
    with pytest.raises(ValueError, match="factory authority"):
        authorized.start(permit, budget, forged)


def test_lifecycle_rejects_reconciliation_from_mutated_authorization() -> None:
    plan, budget, _cleanup_value, authorized, permit = _authorized()
    reservation = _reservation(budget)
    finalizing = authorized.start(
        permit, budget, _commitment(budget, reservation)
    ).begin_finalization()
    reconciliation = _reconciliation(budget, reservation)
    changed_allowances = list(budget.allowances)
    first = changed_allowances[0]
    assert isinstance(first, CappedAllowance)
    changed_allowances[0] = first.model_copy(update={"value": first.value + 1})
    changed_budget = BudgetAuthorization(
        episode_id=budget.episode_id,
        allowances=tuple(changed_allowances),
    )
    payload = reconciliation.model_dump()
    with pytest.raises(ValidationError):
        BudgetReconciliation.model_validate(
            {
                **payload,
                "budget_id": changed_budget.budget_id,
                "authorization_digest": changed_budget.budget_id,
                "authorization": changed_budget,
            }
        )
    assert plan.plan_id == finalizing.plan_id


def test_forged_preflight_seal_cannot_preflight_or_authorize() -> None:
    plan = _plan()
    seal = _seal(plan)
    budget = _budget()
    cleanup = _cleanup_plan()
    planned = EpisodeLifecycle.planned(
        episode_id="episode-1",
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    )
    forged = SealedPreflight.model_construct(**seal.model_dump())
    with pytest.raises(ValueError, match="lacks factory authority"):
        planned.preflight(forged)
    preflighted = planned.preflight(seal)
    with pytest.raises(ValueError, match="lacks factory authority"):
        preflighted.authorize(forged, budget)


def test_forged_cleanup_result_cannot_complete_episode() -> None:
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    cleaning = (
        authorized.start(permit, budget, _commitment(budget, reservation))
        .begin_finalization()
        .finish_finalization(_reconciliation(budget, reservation), _score(plan))
    )
    result = _cleanup(cleanup)
    forged = CleanupResult.model_construct(**result.model_dump())
    with pytest.raises(ValueError, match="lacks factory authority"):
        cleaning.finish_cleanup(forged)


def test_authorized_lifecycle_is_single_use_across_sibling_commitments() -> None:
    import copy

    _plan_value, budget, _cleanup_value, authorized, permit = _authorized()
    first = reserve_budget(
        budget,
        "sibling-a",
        (ResourceAmount(resource="guest_actions", value=10),),
    )
    second = reserve_budget(
        budget,
        "sibling-b",
        (ResourceAmount(resource="guest_actions", value=10),),
    )
    copied = None
    with pytest.raises(TypeError, match="cannot be copied"):
        copied = copy.copy(authorized)
    assert copied is None
    first_running = authorized.start(permit, budget, commit_budget(budget, (first,)))
    assert first_running.state == "running"
    with pytest.raises(ValueError, match="lacks transition authority"):
        authorized.start(permit, budget, commit_budget(budget, (second,)))


def test_forged_side_effect_permit_cannot_start() -> None:
    _plan_value, budget, _cleanup_value, authorized, permit = _authorized()
    reservation = _reservation(budget)
    forged = SideEffectPermit.model_construct(**permit.model_dump())
    with pytest.raises(ValueError, match="lacks factory authority"):
        authorized.start(forged, budget, _commitment(budget, reservation))


def test_finalizing_crash_and_cancel_still_require_cleanup() -> None:
    _plan_value, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    finalizing = authorized.start(
        permit, budget, _commitment(budget, reservation)
    ).begin_finalization()
    crashed = finalizing.crash("judge process exited")
    assert crashed.state == "cleaning"
    assert crashed.finish_cleanup(_cleanup(cleanup)).state == "failed"

    # Each state node is single use only at side-effect start; independent test
    # setup provides a second finalizing authority for cancellation.
    _plan_value, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    finalizing = authorized.start(
        permit, budget, _commitment(budget, reservation)
    ).begin_finalization()
    cancelled = finalizing.cancel("operator cancelled during judging")
    assert cancelled.state == "cleaning"
    assert cancelled.finish_cleanup(_cleanup(cleanup)).state == "cancelled"


def test_cleanup_cannot_succeed_after_timeout() -> None:
    plan = _cleanup_plan()
    late = record_cleanup(
        plan,
        "session",
        status="succeeded",
        evidence_code="late-confirmation",
        duration_ms=plan.actions[0].timeout_ms + 1,
    )
    other = record_cleanup(
        plan,
        "volume",
        status="not_needed",
        evidence_code="not-allocated",
        duration_ms=0,
    )
    with pytest.raises(ValueError, match="after its action timeout"):
        aggregate_cleanup(plan, (late, other))

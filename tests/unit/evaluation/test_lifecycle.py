"""Transition and cleanup contracts for benchmark-neutral evaluation episodes."""

from __future__ import annotations

import importlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

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
    AuthorityModel,
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
_EPISODES = itertools.count(1)


def _plan(episode_id: str = "episode-1") -> DependencyPlan:
    return DependencyPlan(
        release_id="release-1",
        task_id="task-1",
        attempt_id=episode_id,
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


def _budget(episode_id: str = "episode-1") -> BudgetAuthorization:
    return BudgetAuthorization(
        episode_id=episode_id,
        allowances=tuple(
            CappedAllowance(resource=resource, value=100, reporting="required")
            for resource in BUDGET_RESOURCES
        ),
    )


def _cleanup_plan(episode_id: str = "episode-1") -> CleanupPlan:
    return CleanupPlan(
        episode_id=episode_id,
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
        episode_id=plan.attempt_id,
        plan_id=plan.plan_id,
        score_artifact=ArtifactRef(
            sha256=DIGEST,
            media_type="application/json",
            byte_count=100,
        ),
        finalized_at_ms=10,
    )


def _planned() -> tuple[DependencyPlan, BudgetAuthorization, CleanupPlan, EpisodeLifecycle]:
    episode_id = f"episode-{next(_EPISODES)}"
    plan = _plan(episode_id)
    budget = _budget(episode_id)
    cleanup = _cleanup_plan(episode_id)
    return (
        plan,
        budget,
        cleanup,
        EpisodeLifecycle.planned(
            episode_id=episode_id,
            plan_id=plan.plan_id,
            budget_id=budget.budget_id,
            cleanup_plan_id=cleanup.cleanup_plan_id,
        ),
    )


def _authorized() -> tuple[
    DependencyPlan,
    BudgetAuthorization,
    CleanupPlan,
    EpisodeLifecycle,
    SideEffectPermit,
]:
    episode_id = f"episode-{next(_EPISODES)}"
    plan = _plan(episode_id)
    seal = _seal(plan)
    budget = _budget(episode_id)
    cleanup = _cleanup_plan(episode_id)
    episode = EpisodeLifecycle.planned(
        episode_id=episode_id,
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
    parsed = CleanupResult.from_canonical_json(attempted.to_canonical_json())
    with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
        parsed.assert_authority()
    clean = _cleanup(plan)
    assert not clean.rescue_required


def test_permit_cannot_be_constructed_or_deserialized_by_callers() -> None:
    plan, budget, _cleanup_plan_value, _episode, permit = _authorized()
    payload = permit.model_dump()
    for parsed in (
        SideEffectPermit.model_validate(payload),
        SideEffectPermit.from_canonical_json(permit.to_canonical_json()),
    ):
        with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
            parsed.assert_authority()
    with pytest.raises(ValidationError):
        SideEffectPermit(
            episode_id="episode-1",
            plan_id=plan.plan_id,
            preflight_seal_id=DIGEST,
            budget_id=budget.budget_id,
            permit_id=DIGEST,
        )


def test_side_effect_start_is_impossible_before_seal_permit_and_reservation() -> None:
    _plan_value, _budget_value, _cleanup_value, planned = _planned()
    _, budget, _, authorized, permit = _authorized()
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
    parsed = EpisodeLifecycle.from_canonical_json(completed.to_canonical_json())
    with pytest.raises(ValueError, match="lacks transition authority"):
        parsed.begin_finalization()


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
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    reportable = _reconciliation(budget, reservation)
    cleaning = (
        authorized.start(permit, budget, _commitment(budget, reservation))
        .begin_finalization()
        .finish_finalization(reportable, _score(plan))
    )
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
    _plan_value, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    running = authorized.start(permit, budget, _commitment(budget, reservation))
    cancelled = running.cancel("operator requested cancellation")
    assert cancelled.state == "cleaning"
    terminal = cancelled.finish_cleanup(_cleanup(cleanup))
    assert terminal.state == "cancelled"
    assert terminal.cleanup_result_id is not None


def test_preflight_and_infrastructure_failures_are_unscored() -> None:
    plan, budget, cleanup, planned = _planned()
    failed = planned.fail_before_running(kind="preflight", reason="display unavailable")
    assert failed.state == "failed"
    assert failed.score_id is None
    with pytest.raises(ValueError, match="illegal"):
        failed.begin_finalization()
    plan, budget, cleanup, planned = _planned()
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
    if state == "planned":
        _other_plan, _other_budget, _other_cleanup, episode = _planned()
    elif state == "preflighted":
        other_plan, _other_budget, _other_cleanup, other_planned = _planned()
        episode = other_planned.preflight(_seal(other_plan))
    elif state == "authorized":
        episode = authorized
    else:
        running = authorized.start(permit, budget, _commitment(budget, reservation))
        if state == "running":
            episode = running
        elif state == "failed":
            episode = running.crash("crash").finish_cleanup(_cleanup(cleanup))
        elif state == "cancelled":
            episode = running.cancel("cancel").finish_cleanup(_cleanup(cleanup))
        else:
            finalizing = running.begin_finalization()
            if state == "finalizing":
                episode = finalizing
            else:
                cleaning = finalizing.finish_finalization(
                    _reconciliation(budget, reservation), _score(plan)
                )
                episode = (
                    cleaning.finish_cleanup(_cleanup(cleanup)) if state == "completed" else cleaning
                )
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
    for parsed in (
        EpisodeLifecycle.model_validate(payload),
        EpisodeLifecycle.model_validate_json(authorized.to_canonical_json()),
        EpisodeLifecycle.from_canonical_json(authorized.to_canonical_json()),
    ):
        with pytest.raises(ValueError, match="lacks transition authority"):
            parsed.begin_finalization()
    forged = EpisodeLifecycle.model_construct(**payload)
    with pytest.raises(ValueError, match="lacks transition authority"):
        forged.begin_finalization()
    with pytest.raises(ValueError, match="cannot be copied"):
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
    plan, budget, _cleanup_value, planned = _planned()
    seal = _seal(plan)
    forged = SealedPreflight.model_construct(**seal.model_dump())
    with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
        planned.preflight(forged)
    preflighted = planned.preflight(seal)
    with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
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
    with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
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
    with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
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


def test_start_is_atomic_across_threads_and_rolls_back_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    for iteration in range(12):
        _plan_value, budget, _cleanup_value, authorized, permit = _authorized()
        reservations = tuple(
            reserve_budget(
                budget,
                f"race-{iteration}-{suffix}",
                (ResourceAmount(resource="guest_actions", value=10),),
            )
            for suffix in ("a", "b")
        )
        commitments = tuple(commit_budget(budget, (item,)) for item in reservations)
        barrier = Barrier(2)

        def attempt(index: int) -> object:
            barrier.wait()
            try:
                return authorized.start(permit, budget, commitments[index])
            except ValueError as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(attempt, (0, 1)))
        running = [item for item in outcomes if isinstance(item, EpisodeLifecycle)]
        rejected = [item for item in outcomes if isinstance(item, ValueError)]
        assert len(running) == 1
        assert len(rejected) == 1
        assert running[0].state == "running"
        assert "authority" in str(rejected[0])

    _plan_value, budget, _cleanup_value, authorized, permit = _authorized()
    reservation = _reservation(budget)
    commitment = _commitment(budget, reservation)
    original = EpisodeLifecycle._transition
    calls = 0

    def fail_once(self, expected, operation, **updates):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected child construction failure")
        return original(self, expected, operation, **updates)

    monkeypatch.setattr(EpisodeLifecycle, "_transition", fail_once)
    with pytest.raises(ValueError, match="injected child construction failure"):
        authorized.start(permit, budget, commitment)
    assert authorized.start(permit, budget, commitment).state == "running"


def test_finish_cleanup_is_atomic_and_rolls_back_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    cleaning = (
        authorized.start(permit, budget, _commitment(budget, reservation))
        .begin_finalization()
        .finish_finalization(_reconciliation(budget, reservation), _score(plan))
    )
    result = _cleanup(cleanup)
    barrier = Barrier(2)

    def attempt() -> object:
        barrier.wait()
        try:
            return cleaning.finish_cleanup(result)
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: attempt(), (0, 1)))
    terminal = [item for item in outcomes if isinstance(item, EpisodeLifecycle)]
    rejected = [item for item in outcomes if isinstance(item, ValueError)]
    assert len(terminal) == 1
    assert len(rejected) == 1
    assert terminal[0].state == "completed"

    # A cleanup result is single-use even against a separately minted but
    # content-identical cleaning authority.
    plan2, budget2, cleanup2, authorized2, permit2 = _authorized()
    reservation2 = _reservation(budget2)
    cleaning2 = (
        authorized2.start(permit2, budget2, _commitment(budget2, reservation2))
        .begin_finalization()
        .finish_finalization(_reconciliation(budget2, reservation2), _score(plan2))
    )
    with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
        cleaning2.finish_cleanup(result)
    assert cleanup2.episode_id != cleanup.episode_id

    plan3, budget3, cleanup3, authorized3, permit3 = _authorized()
    reservation3 = _reservation(budget3)
    cleaning3 = (
        authorized3.start(permit3, budget3, _commitment(budget3, reservation3))
        .begin_finalization()
        .finish_finalization(_reconciliation(budget3, reservation3), _score(plan3))
    )
    retryable_result = _cleanup(cleanup3)
    original = EpisodeLifecycle._transition
    calls = 0

    def fail_once(self, expected, operation, **updates):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected terminal construction failure")
        return original(self, expected, operation, **updates)

    monkeypatch.setattr(EpisodeLifecycle, "_transition", fail_once)
    with pytest.raises(ValueError, match="injected terminal construction failure"):
        cleaning3.finish_cleanup(retryable_result)
    assert cleaning3.finish_cleanup(retryable_result).state == "completed"


def test_every_lifecycle_parent_is_sequentially_single_use() -> None:
    plan = _plan()
    seal = _seal(plan)
    budget = _budget()
    cleanup = _cleanup_plan()
    planned = EpisodeLifecycle.planned(
        episode_id=budget.episode_id,
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    )
    preflighted = planned.preflight(seal)
    with pytest.raises(ValueError, match="lacks transition authority"):
        planned.preflight(seal)

    authorized, permit = preflighted.authorize(seal, budget)
    with pytest.raises(ValueError, match="lacks transition authority"):
        preflighted.authorize(seal, budget)

    reservation = reserve_budget(
        budget,
        "linear-reservation",
        tuple(ResourceAmount(resource=resource, value=10) for resource in BUDGET_RESOURCES),
    )
    running = authorized.start(permit, budget, commit_budget(budget, (reservation,)))
    finalizing = running.begin_finalization()
    with pytest.raises(ValueError, match="lacks transition authority"):
        running.crash("late sibling")

    cleaning = finalizing.finish_finalization(_reconciliation(budget, reservation), _score(plan))
    with pytest.raises(ValueError, match="lacks transition authority"):
        finalizing.finish_finalization(_reconciliation(budget, reservation), _score(plan))
    terminal = cleaning.finish_cleanup(_cleanup(cleanup))
    assert terminal.state == "completed"
    with pytest.raises(ValueError, match="lacks transition authority"):
        cleaning.finish_cleanup(_cleanup(cleanup))


def test_authorize_and_finalization_edges_are_atomic_under_race() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    plan, budget, _cleanup_value, planned = _planned()
    seal = _seal(plan)
    preflighted = planned.preflight(seal)
    barrier = Barrier(2)

    def authorize() -> object:
        barrier.wait()
        try:
            return preflighted.authorize(seal, budget)
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: authorize(), (0, 1)))
    assert sum(isinstance(item, tuple) for item in outcomes) == 1
    assert sum(isinstance(item, ValueError) for item in outcomes) == 1

    plan, budget, _cleanup_value, authorized, permit = _authorized()
    reservation = _reservation(budget)
    finalizing = authorized.start(
        permit, budget, _commitment(budget, reservation)
    ).begin_finalization()
    reconciliation = _reconciliation(budget, reservation)
    score = _score(plan)
    barrier = Barrier(2)

    def finalize() -> object:
        barrier.wait()
        try:
            return finalizing.finish_finalization(reconciliation, score)
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: finalize(), (0, 1)))
    assert sum(isinstance(item, EpisodeLifecycle) for item in outcomes) == 1
    assert sum(isinstance(item, ValueError) for item in outcomes) == 1


def test_running_competing_edges_and_constructor_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    _plan_value, budget, _cleanup_value, authorized, permit = _authorized()
    reservation = _reservation(budget)
    running = authorized.start(permit, budget, _commitment(budget, reservation))
    barrier = Barrier(2)

    def compete(operation: str) -> object:
        barrier.wait()
        try:
            return (
                running.begin_finalization() if operation == "finalize" else running.crash("boom")
            )
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(compete, ("finalize", "crash")))
    assert sum(isinstance(item, EpisodeLifecycle) for item in outcomes) == 1
    assert sum(isinstance(item, ValueError) for item in outcomes) == 1

    plan = _plan()
    seal = _seal(plan)
    budget = _budget()
    cleanup = _cleanup_plan()
    planned = EpisodeLifecycle.planned(
        episode_id="episode-construction-retry",
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    )
    original = EpisodeLifecycle._transition
    calls = 0

    def fail_once(self, expected, operation, **updates):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("injected edge construction failure")
        return original(self, expected, operation, **updates)

    monkeypatch.setattr(EpisodeLifecycle, "_transition", fail_once)
    with pytest.raises(ValueError, match="injected edge construction failure"):
        planned.preflight(seal)
    assert planned.preflight(seal).state == "preflighted"


def test_reusable_seal_stays_plan_bound_while_episode_authorities_do_not_cross() -> None:
    plan = _plan("shared-plan-attempt")
    seal = _seal(plan)
    cleanup = _cleanup_plan("shared-plan-attempt")
    budgets = (
        _budget(f"seal-reuse-{next(_EPISODES)}"),
        _budget(f"seal-reuse-{next(_EPISODES)}"),
    )
    episodes = tuple(
        EpisodeLifecycle.planned(
            episode_id=budget.episode_id,
            plan_id=plan.plan_id,
            budget_id=budget.budget_id,
            cleanup_plan_id=CleanupPlan(
                episode_id=budget.episode_id,
                actions=cleanup.actions,
            ).cleanup_plan_id,
        ).preflight(seal)
        for budget in budgets
    )
    authorized1, permit1 = episodes[0].authorize(seal, budgets[0])
    authorized2, _permit2 = episodes[1].authorize(seal, budgets[1])
    reservation2 = reserve_budget(
        budgets[1],
        "episode-2-reservation",
        (ResourceAmount(resource="guest_actions", value=1),),
    )
    with pytest.raises(ValueError, match="does not match this episode"):
        authorized2.start(permit1, budgets[1], commit_budget(budgets[1], (reservation2,)))
    assert authorized1.episode_id != authorized2.episode_id


def test_live_episode_lineage_rejects_duplicate_root_and_full_duplicate_path() -> None:
    episode_id = f"lineage-live-{next(_EPISODES)}"
    plan = _plan(episode_id)
    budget = _budget(episode_id)
    cleanup = _cleanup_plan(episode_id)
    root = EpisodeLifecycle.planned(
        episode_id=episode_id,
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    )
    with pytest.raises(ValueError, match="already has a live lineage"):
        EpisodeLifecycle.planned(
            episode_id=episode_id,
            plan_id=plan.plan_id,
            budget_id=budget.budget_id,
            cleanup_plan_id=cleanup.cleanup_plan_id,
        )
    child = root.preflight(_seal(plan))
    with pytest.raises(ValueError, match="already has a live lineage"):
        EpisodeLifecycle.planned(
            episode_id=episode_id,
            plan_id=DIGEST,
            budget_id=budget.budget_id,
            cleanup_plan_id=cleanup.cleanup_plan_id,
        )
    assert child.episode_id == episode_id


def test_episode_root_mint_is_atomic_and_independent_by_identity() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    episode_id = f"lineage-race-{next(_EPISODES)}"
    plan = _plan(episode_id)
    budget = _budget(episode_id)
    cleanup = _cleanup_plan(episode_id)
    barrier = Barrier(2)

    def mint() -> object:
        barrier.wait()
        try:
            return EpisodeLifecycle.planned(
                episode_id=episode_id,
                plan_id=plan.plan_id,
                budget_id=budget.budget_id,
                cleanup_plan_id=cleanup.cleanup_plan_id,
            )
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: mint(), (0, 1)))
    assert sum(isinstance(item, EpisodeLifecycle) for item in outcomes) == 1
    assert sum(isinstance(item, ValueError) for item in outcomes) == 1

    first = f"lineage-independent-{next(_EPISODES)}"
    second = f"lineage-independent-{next(_EPISODES)}"
    roots = tuple(
        EpisodeLifecycle.planned(
            episode_id=value,
            plan_id=_plan(value).plan_id,
            budget_id=_budget(value).budget_id,
            cleanup_plan_id=_cleanup_plan(value).cleanup_plan_id,
        )
        for value in (first, second)
    )
    assert roots[0].episode_id != roots[1].episode_id


def test_unreachable_lineage_releases_weak_registry_lease() -> None:
    import gc
    import weakref

    from local_operator.evaluation.lifecycle import _LIVE_LINEAGES
    from local_operator.evaluation.receipts import _lookup_authority

    episode_id = f"lineage-gc-{next(_EPISODES)}"
    plan = _plan(episode_id)
    budget = _budget(episode_id)
    cleanup = _cleanup_plan(episode_id)
    root = EpisodeLifecycle.planned(
        episode_id=episode_id,
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    )
    child = root.preflight(_seal(plan))
    lineage_ref = weakref.ref(_lookup_authority(child, "episode-lifecycle").lineage)
    assert episode_id in _LIVE_LINEAGES
    del root
    gc.collect()
    assert episode_id in _LIVE_LINEAGES
    del child
    gc.collect()
    assert lineage_ref() is None
    assert episode_id not in _LIVE_LINEAGES
    retry = EpisodeLifecycle.planned(
        episode_id=episode_id,
        plan_id=plan.plan_id,
        budget_id=budget.budget_id,
        cleanup_plan_id=cleanup.cleanup_plan_id,
    )
    assert retry.episode_id == episode_id


def test_lineage_is_private_unforgeable_and_shared_by_children() -> None:
    plan, budget, _cleanup_value, root = _planned()
    child = root.preflight(_seal(plan))
    from local_operator.evaluation.receipts import _lookup_authority

    assert (
        _lookup_authority(root, "episode-lifecycle", allow_consumed=True).lineage
        is _lookup_authority(child, "episode-lifecycle").lineage
    )
    payload = child.model_dump()
    assert "lineage" not in payload
    forged = EpisodeLifecycle.model_construct(**payload)
    with pytest.raises(ValueError, match="lacks transition authority"):
        forged.authorize(_seal(plan), budget)


def test_every_authority_model_blocks_copy_and_constructed_private_injection() -> None:
    import inspect

    from local_operator.evaluation import lifecycle as lifecycle_module
    from local_operator.evaluation import receipts as receipts_module
    from local_operator.evaluation.receipts import AuthorityModel

    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    commitment = _commitment(budget, reservation)
    seal = _seal(plan)
    cleanup_result = _cleanup(cleanup)
    live_models = (authorized, permit, commitment, seal, cleanup_result)
    for live in live_models:
        with pytest.raises(ValueError, match="cannot be copied"):
            live.copy()
        with pytest.raises(ValueError, match="cannot be copied"):
            live.model_copy()
        private_values = {
            name: getattr(live, name, object())
            for name in ("_authority", "_lineage", "_lock", "_consumed", "_receipts")
        }
        construct: Any = type(live).model_construct
        constructed = construct(**live.model_dump(), **private_values)
        assert constructed.__pydantic_private__ in (None, {})
        if isinstance(constructed, EpisodeLifecycle):
            with pytest.raises(ValueError, match="lacks transition authority"):
                constructed.begin_finalization()
        elif hasattr(constructed, "assert_authority"):
            with pytest.raises((ValueError, AttributeError)):
                if isinstance(constructed, BudgetCommitment):
                    constructed.assert_authority(budget)
                else:
                    constructed.assert_authority()

    authority_classes = {
        value
        for module in (receipts_module, lifecycle_module)
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, AuthorityModel)
        and value is not AuthorityModel
    }
    assert authority_classes == {
        SealedPreflight,
        BudgetCommitment,
        CleanupResult,
        SideEffectPermit,
        EpisodeLifecycle,
    }
    assert all(issubclass(value, AuthorityModel) for value in authority_classes)
    assert all(value.copy is AuthorityModel.copy for value in authority_classes)
    assert all(
        value.model_construct.__func__ is AuthorityModel.model_construct.__func__
        for value in authority_classes
    )


def test_constructed_authority_clones_cannot_consume_originals() -> None:
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    commitment = _commitment(budget, reservation)
    cloned_lifecycle = EpisodeLifecycle.model_construct(**authorized.model_dump())
    cloned_permit = SideEffectPermit.model_construct(**permit.model_dump())
    cloned_commitment = BudgetCommitment.model_construct(**commitment.model_dump())
    with pytest.raises(ValueError, match="lacks transition authority"):
        cloned_lifecycle.start(cloned_permit, budget, cloned_commitment)
    running = authorized.start(permit, budget, commitment)
    finalizing = running.begin_finalization()
    cleaning = finalizing.finish_finalization(_reconciliation(budget, reservation), _score(plan))
    result = _cleanup(cleanup)
    cloned_result = CleanupResult.model_construct(**result.model_dump())
    with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
        cleaning.finish_cleanup(cloned_result)
    assert cleaning.finish_cleanup(result).state == "completed"


def test_validation_context_never_mints_authority_for_any_model() -> None:
    from pydantic import TypeAdapter

    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    live_models = (
        _seal(plan),
        _commitment(budget, reservation),
        _cleanup(cleanup),
        permit,
        authorized,
    )
    hostile_context = {
        "preflight_seal_factory": object(),
        "preflight_receipts": object(),
        "budget_commitment_factory": object(),
        "cleanup_result_factory": object(),
        "cleanup_receipts": object(),
        "permit_factory": object(),
        "lifecycle_factory": object(),
        "episode_lineage": object(),
        "_FACTORY_TOKEN": object(),
        "_LIFECYCLE_TOKEN": object(),
    }
    for live in live_models:
        adapter = TypeAdapter(type(live))
        payload = live.model_dump(mode="json")
        encoded = live.to_canonical_json()
        parsed_values = (
            type(live).model_validate(payload, context=hostile_context),
            type(live).model_validate_json(encoded, context=hostile_context),
            adapter.validate_python(payload, context=hostile_context),
            adapter.validate_json(encoded, context=hostile_context),
        )
        for parsed in parsed_values:
            assert parsed.__pydantic_private__ in (None, {})
            if isinstance(parsed, EpisodeLifecycle):
                with pytest.raises(ValueError, match="lacks transition authority"):
                    parsed.begin_finalization()
            elif isinstance(parsed, BudgetCommitment):
                with pytest.raises(
                    ValueError, match="lacks factory authority|process-local authority"
                ):
                    parsed.assert_authority(budget)
            else:
                with pytest.raises(
                    ValueError, match="lacks factory authority|process-local authority"
                ):
                    parsed.assert_authority()


def test_authority_validators_do_not_read_context_or_factory_tokens() -> None:
    import inspect

    from local_operator.evaluation import lifecycle as lifecycle_module
    from local_operator.evaluation import receipts as receipts_module

    source = inspect.getsource(receipts_module) + inspect.getsource(lifecycle_module)
    assert "ValidationInfo" not in source
    assert ".context" not in source
    assert "_FACTORY_TOKEN" not in source
    assert "_LIFECYCLE_TOKEN" not in source
    assert "context=" not in source


def test_factory_objects_retain_live_receipts_lineage_and_authority() -> None:
    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    seal = _seal(plan)
    commitment = _commitment(budget, reservation)
    result = _cleanup(cleanup)
    from local_operator.evaluation.receipts import _lookup_authority

    records = (
        _lookup_authority(seal, "preflight-seal"),
        _lookup_authority(commitment, "budget-commitment"),
        _lookup_authority(result, "cleanup-result"),
        _lookup_authority(permit, "side-effect-permit"),
        _lookup_authority(authorized, "episode-lifecycle"),
    )
    assert all(
        live.__pydantic_private__ in (None, {})
        for live in (seal, commitment, result, permit, authorized)
    )
    assert records[0].receipts
    assert records[2].receipts
    assert records[4].lineage.episode_id == authorized.episode_id


def test_authorized_failure_revokes_permit_and_wrong_permit_is_retryable() -> None:
    plan, budget, _cleanup_value, authorized, permit = _authorized()
    _other_plan, _other_budget, _other_cleanup, _other_authorized, wrong = _authorized()
    with pytest.raises(ValueError, match="does not match this episode"):
        authorized.fail_before_running(
            kind="infrastructure", reason="allocator unavailable", permit=wrong
        )
    permit.assert_authority()
    failed = authorized.fail_before_running(
        kind="infrastructure", reason="allocator unavailable", permit=permit
    )
    assert failed.state == "failed"
    with pytest.raises(ValueError, match="lacks factory authority|process-local authority"):
        permit.assert_authority()
    assert failed.score_id is None
    assert plan.plan_id == failed.plan_id


def test_authorized_fail_and_start_race_has_exactly_one_child() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    _plan_value, budget, _cleanup_value, authorized, permit = _authorized()
    reservation = _reservation(budget)
    commitment = _commitment(budget, reservation)
    barrier = Barrier(2)

    def attempt(operation: str) -> object:
        barrier.wait()
        try:
            if operation == "start":
                return authorized.start(permit, budget, commitment)
            return authorized.fail_before_running(
                kind="infrastructure", reason="allocator unavailable", permit=permit
            )
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(attempt, ("start", "fail")))
    children = [item for item in outcomes if isinstance(item, EpisodeLifecycle)]
    errors = [item for item in outcomes if isinstance(item, ValueError)]
    assert len(children) == 1
    assert len(errors) == 1
    assert children[0].state in ("running", "failed")
    assert "authority" in str(errors[0])


def test_base_model_copy_paths_never_copy_registry_authority() -> None:
    from pydantic import BaseModel

    from local_operator.evaluation.receipts import _lookup_authority

    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    originals = (
        _seal(plan),
        _commitment(budget, reservation),
        _cleanup(cleanup),
        permit,
        authorized,
    )
    for original in originals:
        with pytest.raises(TypeError, match="cannot be copied"):
            BaseModel.model_copy(original)
        clones = (
            BaseModel.copy(original),
            BaseModel.__copy__(original),
            BaseModel.__deepcopy__(original),
        )
        for clone in clones:
            clone = cast(AuthorityModel, clone)
            assert clone is not original
            with pytest.raises(ValueError, match="process-local authority"):
                _lookup_authority(clone, "episode-lifecycle")
            assert clone.__pydantic_private__ in (None, {})
        # TypeAdapter may return the same already-validated instance; this is
        # aliasing and still leaves only one consumable registry identity.
        from pydantic import TypeAdapter

        assert TypeAdapter(type(original)).validate_python(original) is original


def test_registry_is_identity_keyed_and_weakly_releases_retained_receipts() -> None:
    import gc
    import weakref

    from pydantic import BaseModel

    from local_operator.evaluation.receipts import (
        _authority_registry_size,
        _lookup_authority,
    )

    plan = _plan(f"registry-gc-{next(_EPISODES)}")
    seal = _seal(plan)
    receipt_ref = weakref.ref(_lookup_authority(seal, "preflight-seal").receipts[0])
    baseline = _authority_registry_size()
    clone = cast(SealedPreflight, BaseModel.copy(seal))
    assert clone == seal and clone is not seal
    with pytest.raises(ValueError, match="process-local authority"):
        _lookup_authority(clone, "preflight-seal")
    del clone
    del seal
    gc.collect()
    assert receipt_ref() is None
    assert _authority_registry_size() < baseline


def test_registry_dead_callback_cannot_remove_reused_identity_entry() -> None:
    import weakref

    from local_operator.evaluation.receipts import (
        _AUTHORITY_REGISTRY,
        _AUTHORITY_REGISTRY_LOCK,
        AuthorityRecord,
        _remove_authority,
    )

    first = _seal(_plan(f"callback-first-{next(_EPISODES)}"))
    second = _seal(_plan(f"callback-second-{next(_EPISODES)}"))
    stale_reference: Any = weakref.ref(first)
    current_reference: Any = weakref.ref(second)
    synthetic_id = -1
    with _AUTHORITY_REGISTRY_LOCK:
        _AUTHORITY_REGISTRY[synthetic_id] = (
            current_reference,
            AuthorityRecord(kind="preflight-seal"),
        )
    _remove_authority(synthetic_id, stale_reference)
    with _AUTHORITY_REGISTRY_LOCK:
        assert _AUTHORITY_REGISTRY[synthetic_id][0] is current_reference
        del _AUTHORITY_REGISTRY[synthetic_id]


def test_authority_models_store_no_private_capabilities_or_serialized_state() -> None:
    from local_operator.evaluation.receipts import AuthorityModel

    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    models = (
        _seal(plan),
        _commitment(budget, reservation),
        _cleanup(cleanup),
        permit,
        authorized,
    )
    forbidden = ("authority", "lineage", "lock", "consumed", "receipts")
    for model in models:
        assert isinstance(model, AuthorityModel)
        assert model.__pydantic_private__ in (None, {})
        canonical = model.to_canonical_json().decode()
        assert not any(name in canonical for name in forbidden)


def test_base_copy_duplicate_path_rejects_and_originals_complete_once() -> None:
    from pydantic import BaseModel

    plan, budget, cleanup, authorized, permit = _authorized()
    reservation = _reservation(budget)
    commitment = _commitment(budget, reservation)
    cloned_lifecycle = cast(EpisodeLifecycle, BaseModel.copy(authorized))
    cloned_permit = cast(SideEffectPermit, BaseModel.copy(permit))
    cloned_commitment = cast(BudgetCommitment, BaseModel.copy(commitment))
    with pytest.raises(ValueError, match="transition authority|process-local authority"):
        cloned_lifecycle.start(cloned_permit, budget, cloned_commitment)
    cleaning = (
        authorized.start(permit, budget, commitment)
        .begin_finalization()
        .finish_finalization(_reconciliation(budget, reservation), _score(plan))
    )
    result = _cleanup(cleanup)
    cloned_result = cast(CleanupResult, BaseModel.copy(result))
    with pytest.raises(ValueError, match="process-local authority"):
        cleaning.finish_cleanup(cloned_result)
    assert cleaning.finish_cleanup(result).state == "completed"

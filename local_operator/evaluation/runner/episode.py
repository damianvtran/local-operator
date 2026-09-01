"""Run one evaluation episode and leave behind exactly one verifiable bundle.

This module composes the four released foundations and adds no new invariants
of its own:

* ``protocol``/``adapters.api`` -- the observation and action wire shapes.
* ``adapters.supervisor`` -- ``VerifiedAdapterSession``, which is parent-owned
  truth about what the adapter did, plus durable rescue.
* ``receipts``/``lifecycle`` -- preflight, budget, cleanup and the episode
  state machine, all of them single-use process-local authorities.
* ``evidence`` -- the append-only journal whose verifier is the acceptance
  test for everything written here.

**Import isolation.** Nothing here may import providers, config, tools, the
TUI, mobile, or session code; ``tests/unit/evaluation/runner/test_isolation.py``
asserts it. An episode must be reproducible from its pinned inputs, and a
transitive import of session configuration is how a benchmark result silently
starts depending on the operator's own settings. The provider-backed model
client lives in ``provider_client.py`` for exactly this reason.

**Concurrency.** One runner drives one episode, one worker, and one writer, and
shares no mutable state. ``lifecycle._EpisodeLineage`` is process-global keyed
by ``episode_id``, so episodes running in parallel in one process MUST use
distinct ``episode_id`` values or they will contend for the same lineage lease.
Run-level budget aggregation across episodes is deliberately out of scope.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from local_operator.evaluation.adapters.api import (
    AskUserExchangeParams,
    CleanupParams,
    CloseParams,
    ExecuteParams,
    Handshake,
    InspectRequirementsParams,
    PrepareParams,
    RescueDescriptor,
    ResetStartParams,
    ScopedInfraValue,
    ScoreParams,
    SecretRef,
)
from local_operator.evaluation.adapters.supervisor import (
    AdapterSupervisor,
    HostVerifier,
    VerifiedAdapterSession,
    persist_rescue,
    run_rescue,
)
from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    BudgetCommitmentPayload,
    CancelPayload,
    CleanupPayload,
    EnvironmentStepPayload,
    ErrorPayload,
    EvidenceManifest,
    FinalizationIntent,
    LifecycleTransitionPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    ObservationPayload,
    OutcomeDraft,
    PreflightPayload,
    ReconciliationPayload,
    RouteIdentity,
    ScoreArtifact,
    ScoringResultPayload,
    UsageCostPayload,
    UserSimulatorExchangePayload,
    canonical_digest,
)
from local_operator.evaluation.evidence.store import EvidenceError, EvidenceWriter
from local_operator.evaluation.lifecycle import (
    CleanupAction,
    CleanupPlan,
    CleanupReceipt,
    EpisodeLifecycle,
    aggregate_cleanup,
    plan_episode,
)
from local_operator.evaluation.protocol import (
    ActionBatch,
    AskUserAction,
    FinishAction,
    Observation,
)
from local_operator.evaluation.receipts import (
    AvailableUsage,
    BudgetAuthorization,
    BudgetReservation,
    DependencyPlan,
    RedactionSet,
    SealedPreflight,
    UnavailableUsage,
    Usage,
    commit_budget,
    reconcile_budget,
)
from local_operator.evaluation.runner.model import EpisodeModelClient
from local_operator.evaluation.runner.responder import NullUserResponder, UserResponder

# Cleanup kinds are a closed vocabulary in ``lifecycle.CleanupAction``. A worker
# session is the one resource the parent always knows exists before the adapter
# has described anything, which is what makes the provisional plan expressible.
_PROVISIONAL_CLEANUP_ACTION = "close-session"

EpisodeStatus = Literal[
    "completed",
    "failed",
    "cancelled",
    "abandoned",
    "failed_pre_bundle",
]


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything pinned about one episode before anything is launched.

    These values identify the run in the evidence manifest, so they are inputs
    rather than anything the adapter may assert: an adapter that could choose
    its own ``task_digest`` could make two different tasks look like one result.
    """

    episode_id: str
    task_id: str
    task_digest: str
    input_digest: str
    benchmark_id: str
    benchmark_release: str
    environment_digest: str
    environment_release: str
    config_digest: str
    harness_version: str
    harness_git_revision: str
    requested_route: RouteIdentity
    dependency_plan: DependencyPlan
    budget: BudgetAuthorization
    preflight: SealedPreflight
    reservations: tuple[BudgetReservation, ...]
    fallback_policy: Literal["forbid", "allow_compatible", "allow_any"] = "forbid"
    secret_refs: tuple[SecretRef, ...] = ()
    infra_values: tuple[ScopedInfraValue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeConfig:
    """Operational limits that shape execution without changing its meaning.

    Timeouts are per adapter call. ``max_steps`` bounds the step loop; reaching
    it is a truncation, which is a normal scored outcome rather than a failure,
    so it must not be conflated with the budget's own limits.
    """

    evidence_root: Path
    artifact_root: Path
    rescue_root: Path
    max_steps: int = 50
    prepare_timeout: float = 120.0
    reset_timeout: float = 120.0
    step_timeout: float = 120.0
    score_timeout: float = 120.0
    cleanup_timeout: float = 60.0
    ask_deadline_ms: int = 60_000
    handshake_timeout: float = 30.0


@dataclass(frozen=True)
class EpisodeOutcome:
    """What happened, and where the evidence for it lives.

    ``bundle_root`` is ``None`` only for ``failed_pre_bundle``: the manifest
    needs identifiers that do not exist until ``prepare`` has returned, so a
    failure before that point has nowhere to write evidence.
    """

    status: EpisodeStatus
    episode_id: str
    bundle_root: Path | None
    score: ScoreArtifact | None = None
    reportability_label: str | None = None
    comparability_label: str | None = None
    rescue_required: bool = False
    rescue_complete: bool | None = None
    diagnostic: str | None = None


class _Cancelled(Exception):
    """Raised inside the step loop to unwind to the cancellation terminal."""


class _EvidenceFailure(Exception):
    """Raised when the writer itself failed and the bundle can only be abandoned."""


class EpisodeRunner:
    """Drives one episode from launch to a sealed or abandoned bundle."""

    def __init__(
        self,
        spec: EpisodeSpec,
        config: EpisodeConfig,
        *,
        selector: Any,
        model: EpisodeModelClient,
        responder: UserResponder | None = None,
        redactions: RedactionSet | None = None,
        launch: Any = AdapterSupervisor.launch,
        rescue: Any = run_rescue,
    ) -> None:
        self._spec = spec
        self._config = config
        self._selector = selector
        self._model = model
        self._responder = responder or NullUserResponder()
        self._redactions = redactions or RedactionSet.from_resolved_values(())
        self._launch = launch
        self._rescue = rescue

        self._writer: EvidenceWriter | None = None
        self._session: VerifiedAdapterSession | None = None
        self._supervisor: Any = None
        self._descriptor: RescueDescriptor | None = None
        self._cleanup_plan: CleanupPlan | None = None
        self._lifecycle: EpisodeLifecycle | None = None
        self._rescue_required = False
        self._transcript: list[Observation] = []
        self._usage_totals: dict[str, int] = {}
        self._provider_cost_micros = 0
        self._model_cycles = 0
        self._guest_actions = 0
        self._simulator_turns = 0
        self._last_exchange_id: str | None = None
        self._steps_taken = 0
        self._truncated = False
        self._last_step_terminated = False
        self._finalization_id = f"final-{spec.episode_id}"
        self._scoring_operation_id = f"score-{spec.episode_id}"
        self._started_ms = _now_ms()

    async def run(self) -> EpisodeOutcome:
        """Execute the episode; every exit path leaves a verifiable bundle."""

        try:
            handshake = await self._launch_and_prepare()
        except _EvidenceFailure as error:
            return await self._abandon_for_evidence(str(error))
        except BaseException as error:
            # No manifest is possible yet: it must carry the cleanup plan ID
            # that ``prepare`` returns. There is nothing to write evidence to,
            # so the only obligations are rescue and reaping.
            await self._emergency_teardown()
            return EpisodeOutcome(
                status="failed_pre_bundle",
                episode_id=self._spec.episode_id,
                bundle_root=None,
                rescue_required=self._rescue_required,
                diagnostic=_diagnostic(error),
            )
        try:
            return await self._run_with_bundle(handshake)
        except _EvidenceFailure as error:
            return await self._abandon_for_evidence(str(error))

    # ------------------------------------------------------------------
    # Launch, prepare, and the two-stage rescue persistence
    # ------------------------------------------------------------------

    async def _launch_and_prepare(self) -> Handshake:
        """Bring up the worker and prepare the environment.

        ADAPTER CONTRACT: ``prepare`` must not create environment resources;
        ``reset_start`` does. The whole two-stage persistence below depends on
        it, and an adapter that allocates during ``prepare`` can leak a
        resource that no persisted descriptor names.
        """

        supervisor = self._launch(self._selector)
        self._supervisor = supervisor
        handshake = await supervisor.handshake(timeout=self._config.handshake_timeout)
        verifier = HostVerifier(
            self._spec.task_id,
            self._spec.episode_id,
            self._config.artifact_root,
        )
        session = VerifiedAdapterSession(
            supervisor,
            verifier,
            rescue_required=self._mark_rescue_required,
        )
        self._session = session

        await session.inspect_requirements(
            InspectRequirementsParams(),
            timeout=self._config.prepare_timeout,
        )

        # STAGE 1 of two-stage rescue persistence.
        #
        # ``VerifiedAdapterSession.prepare`` refuses to run without a persisted
        # rescue descriptor, but a descriptor embeds the cleanup plan that only
        # ``prepare`` can return. The circularity is resolved by persisting a
        # PROVISIONAL descriptor first, carrying the one action the parent can
        # author unaided: close this worker's session. That is sound precisely
        # because ``prepare`` is declarative -- if it creates nothing, the
        # provisional plan covers everything that could exist at this instant.
        provisional = CleanupPlan(
            episode_id=self._spec.episode_id,
            actions=(
                CleanupAction(
                    action_id=_PROVISIONAL_CLEANUP_ACTION,
                    kind="close_session",
                    resource_ref=self._spec.episode_id,
                    timeout_ms=int(self._config.cleanup_timeout * 1000),
                    max_attempts=2,
                ),
            ),
        )
        self._persist_descriptor(handshake, provisional)

        result = await session.prepare(
            PrepareParams(
                operation_id=f"prepare-{self._spec.episode_id}",
                episode_id=self._spec.episode_id,
                secret_refs=self._spec.secret_refs,
                infra_values=self._spec.infra_values,
            ),
            timeout=self._config.prepare_timeout,
        )

        # STAGE 2, before any side effect exists.
        #
        # ``reset_start`` is the side-effect boundary, so the real plan must be
        # durable before it is called and not merely before cleanup. Re-persist
        # now: from here on a crashed parent can rescue every resource the
        # adapter is about to create.
        self._cleanup_plan = result.cleanup_plan
        self._persist_descriptor(handshake, result.cleanup_plan)
        return handshake

    def _persist_descriptor(self, handshake: Handshake, plan: CleanupPlan) -> None:
        descriptor = RescueDescriptor(
            schema_version="1.0",
            selector=self._selector,
            handshake=handshake,
            episode_id=self._spec.episode_id,
            cleanup_plan=plan,
            secret_refs=self._spec.secret_refs,
            infra_values=self._spec.infra_values,
            artifact_root=str(self._config.artifact_root),
        )
        persist_rescue(self._config.rescue_root, descriptor)
        self._descriptor = descriptor
        session = self._session
        if session is not None:
            session.mark_rescue_persisted(descriptor.descriptor_id)

    def _mark_rescue_required(self) -> None:
        self._rescue_required = True

    # ------------------------------------------------------------------
    # Bundle-backed execution
    # ------------------------------------------------------------------

    async def _run_with_bundle(self, handshake: Handshake) -> EpisodeOutcome:
        plan = self._cleanup_plan
        assert plan is not None
        manifest = self._build_manifest(handshake, plan)
        root = self._config.evidence_root / self._spec.episode_id
        try:
            writer = EvidenceWriter.create(root, manifest, self._redactions)
        except EvidenceError as error:
            # The bundle never opened, so there is nothing to abandon.
            await self._emergency_teardown()
            return EpisodeOutcome(
                status="failed_pre_bundle",
                episode_id=self._spec.episode_id,
                bundle_root=None,
                rescue_required=self._rescue_required,
                diagnostic=_diagnostic(error),
            )
        self._writer = writer
        try:
            return await self._execute(handshake)
        finally:
            writer.close()

    async def _execute(self, handshake: Handshake) -> EpisodeOutcome:
        spec = self._spec

        lifecycle = plan_episode(
            episode_id=spec.episode_id,
            plan_id=spec.dependency_plan.plan_id,
            budget_id=spec.budget.budget_id,
            cleanup_plan_id=self._require_plan().cleanup_plan_id,
        )

        # Preflight is event #0 and the commitment must precede every piece of
        # execution evidence; both the writer's phase check and the verifier
        # enforce this ordering independently.
        self._append(
            "preflight",
            PreflightPayload(
                sealed_preflight_id=spec.preflight.seal_id,
                plan_id=spec.dependency_plan.plan_id,
                receipt_ids=tuple(spec.preflight.receipt_digests),
                passed=spec.preflight.successful,
            ),
        )
        lifecycle = lifecycle.preflight(spec.preflight)
        lifecycle, permit = lifecycle.authorize(spec.preflight, spec.budget)
        commitment = commit_budget(spec.budget, spec.reservations)
        self._append(
            "budget_commitment",
            BudgetCommitmentPayload(
                commitment_id=commitment.commitment_id,
                budget_id=spec.budget.budget_id,
                reservation_ids=tuple(commitment.reservation_ids),
                reserved_summary_digest=canonical_digest(
                    "runner-reserved-summary-v1",
                    [amount.model_dump(mode="json") for amount in commitment.reserved],
                ),
            ),
        )
        lifecycle = lifecycle.start(permit, spec.budget, commitment)
        self._lifecycle = lifecycle
        self._append_lifecycle(lifecycle, state="running")

        try:
            await self._reset_and_observe()
            await self._step_loop()
        except _Cancelled as cancel:
            return await self._finalize_cancelled(str(cancel) or "cancelled")
        except _EvidenceFailure:
            raise
        except EvidenceError as error:
            # A writer error is never an episode failure: the journal itself is
            # unusable, so finalizing would write into a poisoned bundle. Route
            # it to the abandonment path like any other evidence failure.
            raise _EvidenceFailure(_diagnostic(error)) from error
        except BaseException as error:
            return await self._finalize_failure(error)
        return await self._finalize_scored(handshake)

    async def _reset_and_observe(self) -> None:
        session = self._require_session()
        result = await session.reset_start(
            ResetStartParams(
                operation_id=f"reset-{self._spec.episode_id}",
                task_id=self._spec.task_id,
                episode_id=self._spec.episode_id,
            ),
            timeout=self._config.reset_timeout,
        )
        self._record_observation(result.observation)

    def _record_observation(self, observation: Observation) -> None:
        """Publish an observation's frames, then its event.

        Artifacts are published FIRST because the verifier resolves every
        reference an event makes; an observation event naming bytes that are
        not yet in the bundle is an invalid bundle, not a race to be tolerated.
        """

        artifacts = []
        for frame in observation.frames:
            data = _read_artifact(self._config.artifact_root, frame.artifact.sha256)
            artifacts.append(
                self._publish(
                    data,
                    media_type=frame.artifact.media_type,
                    expected_sha256=frame.artifact.sha256,
                )
            )
        text_artifact = None
        if observation.text:
            text_artifact = self._publish(observation.text.encode("utf-8"), media_type="text/plain")
        self._append(
            "observation",
            ObservationPayload(
                observation_id=observation.observation_id,
                sequence=observation.sequence,
                artifacts=tuple(artifacts),
                text_artifact=text_artifact,
            ),
        )
        self._transcript.append(observation)

    async def _step_loop(self) -> None:
        session = self._require_session()
        while True:
            current = session.verifier.current_observation
            assert current is not None
            if self._steps_taken >= self._config.max_steps:
                # Truncation is not a failure: the episode is scored on the
                # state it reached. It is recorded on the LAST step rather than
                # as a new event, which is why it must be applied before the
                # step is written -- see ``_execute_batch``.
                self._truncated = True
                return
            decision = await self._decide(current)
            batch = decision.action_batch
            terminal = _terminal_kind(batch)
            if terminal == "finish":
                self._append_batch(batch, terminal="finish")
                return
            if terminal == "ask_user":
                await self._run_ask(batch)
                continue
            await self._execute_batch(batch)
            if self._last_step_terminated:
                return

    async def _decide(self, observation: Observation) -> Any:
        """Ask the model, writing request/response/usage in that exact order.

        The verifier binds a response to its request and a usage record to that
        response, so all three must be written even when the provider reports
        nothing -- a missing usage record leaves an unclosed operation and the
        bundle cannot reach a terminal.
        """

        request_id = f"req-{self._model_cycles}-{uuid.uuid4().hex[:12]}"
        message_count = len(self._transcript)
        # The provider is called BEFORE the request event is written, even
        # though the three events keep their required request/response/usage
        # order in the journal. A request written first would be left unclosed
        # by a provider failure, and the verifier requires every request to
        # carry its response and usage before any terminal (verify.py:760) --
        # so recording it eagerly would make the failure path unsealable.
        try:
            decision = await self._model.decide(observation, tuple(self._transcript))
        except _EvidenceFailure:
            raise
        except BaseException as error:
            raise _ProviderFailure(_diagnostic(error)) from error
        self._append(
            "model_request",
            ModelRequestPayload(
                request_id=request_id,
                requested_route=self._spec.requested_route,
                tool_schema_digest=canonical_digest(
                    "runner-tool-schema-v1", {"episode_id": self._spec.episode_id}
                ),
                input_tokens=decision.usage.input_tokens,
                message_count=message_count,
                tool_count=0,
            ),
        )
        self._append(
            "model_response",
            ModelResponsePayload(
                request_id=request_id,
                provider_request_id=decision.provider_request_id,
                requested_route=self._spec.requested_route,
                served_route=decision.route,
                stop_reason=decision.stop_reason,
                output_tokens=decision.usage.output_tokens,
                reasoning_tokens=decision.usage.reasoning_tokens,
                cache_read_tokens=decision.usage.cache_read_tokens,
                cache_write_tokens=decision.usage.cache_write_tokens,
                tool_call_count=decision.tool_call_count,
            ),
        )
        self._append(
            "usage_cost",
            UsageCostPayload(
                request_id=request_id,
                input_tokens=decision.usage.input_tokens,
                output_tokens=decision.usage.output_tokens,
                reasoning_tokens=decision.usage.reasoning_tokens,
                cache_read_tokens=decision.usage.cache_read_tokens,
                cache_write_tokens=decision.usage.cache_write_tokens,
                cost_microusd=decision.cost_micros,
            ),
        )
        self._model_cycles += 1
        self._provider_cost_micros += decision.cost_micros
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            self._usage_totals[name] = self._usage_totals.get(name, 0) + getattr(
                decision.usage, name
            )
        return decision

    def _append_batch(
        self, batch: ActionBatch, *, terminal: Literal["finish", "ask_user"] | None
    ) -> None:
        canonical = batch.to_canonical_json()
        artifact = self._publish(canonical, media_type="application/json")
        self._append(
            "action_batch",
            ActionBatchPayload(
                action_batch_id=_batch_id(batch),
                observation_id=batch.observation_id,
                action_count=len(batch.actions),
                action_artifact=artifact,
                terminal=terminal,
            ),
        )

    async def _execute_batch(self, batch: ActionBatch) -> None:
        """Execute a batch, then write its batch and step evidence.

        The adapter is called BEFORE the ``action_batch`` event is written. The
        journal still carries batch-then-step in the required order, but a
        non-terminal batch that never produced a step is a permanently invalid
        bundle (verify.py:762-766 requires every non-finish batch to have its
        step), so a crash inside ``execute`` must not leave one behind.
        """

        session = self._require_session()
        params = ExecuteParams(
            operation_id=f"exec-{self._steps_taken}-{uuid.uuid4().hex[:12]}",
            action_batch=batch,
            action_batch_id=_adapter_batch_id(batch),
        )
        result = await session.execute(params, timeout=self._config.step_timeout)
        self._append_batch(batch, terminal=None)
        self._steps_taken += 1
        self._guest_actions += len(batch.actions)
        truncated = self._truncated or self._steps_taken >= self._config.max_steps
        # The step event MUST precede its output observation: the verifier
        # expects the observation it just declared, and reversing the two makes
        # the observation unbound.
        self._append(
            "environment_step",
            EnvironmentStepPayload(
                step_id=result.receipt.operation_id,
                action_batch_id=_batch_id(batch),
                receipt_id=result.receipt.receipt_id,
                input_observation_id=result.receipt.input_observation_id,
                output_observation_id=result.receipt.output_observation_id,
                terminated=False,
                truncated=truncated,
            ),
        )
        self._truncated = truncated
        self._last_step_terminated = truncated
        self._record_observation(result.observation)

    async def _run_ask(self, batch: ActionBatch) -> None:
        """Answer an ask, then execute the ask batch like any other.

        The choreography is forced by the verifier's one-batch-per-observation
        rule (verify.py:575-580): the ask cannot be a batch that is written and
        then abandoned, so the answer is obtained FIRST and the ask batch is
        then executed normally, producing the observation the model sees next.
        """

        session = self._require_session()
        # The adapter exchange is keyed by the model's own request ID so the
        # answer is traceable back to the exact action that asked for it.
        ask_id, prompt = _ask_prompt(batch)
        begin = AskUserExchangeParams(
            operation_id=f"ask-begin-{ask_id}",
            episode_id=self._spec.episode_id,
            ask_id=ask_id,
            prompt=prompt,
        )
        session.begin_ask(begin)
        answer = await self._responder.ask(prompt, self._config.ask_deadline_ms)
        if answer is None:
            # An outstanding ask may not be left open, and there is no way to
            # withdraw one. Cancelling is the only coherent resolution.
            raise _Cancelled("ask-user was not answered")
        finish = begin.model_copy(update={"operation_id": f"ask-finish-{ask_id}", "answer": answer})
        result = await session.finish_ask(finish, timeout=self._config.step_timeout)
        request_artifact = self._publish(prompt.encode("utf-8"), media_type="text/plain")
        response_artifact = self._publish(answer.encode("utf-8"), media_type="text/plain")
        exchange_id = ask_id
        self._append(
            "user_simulator_exchange",
            UserSimulatorExchangePayload(
                exchange_id=exchange_id,
                previous_exchange_id=self._last_exchange_id,
                request_artifact=request_artifact,
                response_artifact=response_artifact,
                receipt_id=canonical_digest(
                    "runner-ask-receipt-v1",
                    {"ask_id": ask_id, "accepted": result.accepted},
                ),
            ),
        )
        self._last_exchange_id = exchange_id
        self._simulator_turns += 1
        # The ask batch is then executed like any other batch: a lone
        # AskUserAction is a legal batch, and executing it is what produces the
        # observation carrying the answer's effect, which the model sees next.
        await self._execute_batch(batch)

    # ------------------------------------------------------------------
    # Terminal paths
    # ------------------------------------------------------------------

    async def _finalize_scored(self, handshake: Handshake) -> EpisodeOutcome:
        session = self._require_session()
        intent = FinalizationIntent(
            kind="score",
            scorer_id=handshake.metadata.adapter_id,
            # The scorer identity pins the adapter build that graded this run.
            # ``StrictIdentifier`` forbids "+", so the digest is joined with a
            # "-": the value only has to be exact and stable, not semver.
            scorer_version=(
                f"{handshake.metadata.version}-{handshake.metadata.package_digest[:16]}"
            ),
        )
        self._begin_finalization(intent, self._scoring_operation_id)
        try:
            result = await session.score(
                ScoreParams(
                    operation_id=self._scoring_operation_id,
                    episode_id=self._spec.episode_id,
                ),
                timeout=self._config.score_timeout,
            )
        except _EvidenceFailure:
            raise
        except BaseException as error:
            # ADAPTER CONTRACT: ``score()`` returns a SCORED artifact or raises.
            # "Unscored" is a harness decision expressed through the
            # finalization intent, never an adapter return value.
            #
            # ``scoring_start`` is already durable and a scored-intent
            # finalization cannot seal unscored, so no legal terminal remains.
            # Rescue first (the worker may be poisoned), then abandon.
            return await self._abandon_after_scoring_failure(error)
        score = result.score
        self._append_receipt(
            "scoring_result",
            ScoringResultPayload(
                finalization_id=self._finalization_id,
                scoring_operation_id=self._scoring_operation_id,
                score=score,
            ),
        )
        return await self._close_out(score, failure_kind=None, cancelled=False)

    async def _finalize_failure(self, error: BaseException) -> EpisodeOutcome:
        """Finalize unscored after a mid-episode failure.

        Both branches begin finalization first: ``record_final_lifecycle``
        refuses to write without the finalizing marker, so a crash path that
        skips it has no terminal at all and the bundle can only be abandoned.
        """

        provider_failure = isinstance(error, _ProviderFailure)
        category: Literal["adapter", "provider"] = "provider" if provider_failure else "adapter"
        self._append(
            "error",
            ErrorPayload(
                error_id=f"err-{uuid.uuid4().hex[:12]}",
                category=category,
                diagnostic_code=_diagnostic_code(error),
                retryable=False,
            ),
        )
        reason: Literal["crash", "infrastructure_failure"] = (
            "infrastructure_failure" if provider_failure else "crash"
        )
        self._begin_finalization(FinalizationIntent(kind="unscored"), None)
        score = ScoreArtifact(status="unscored", reason=reason)
        return await self._close_out(
            score,
            failure_kind="crash" if not provider_failure else "infrastructure",
            cancelled=False,
            diagnostic=_diagnostic(error),
        )

    async def _finalize_cancelled(self, reason: str) -> EpisodeOutcome:
        self._append(
            "cancel",
            CancelPayload(
                cancellation_id=f"cancel-{uuid.uuid4().hex[:12]}",
                source="harness",
                diagnostic_code="cancelled",
            ),
        )
        self._begin_finalization(FinalizationIntent(kind="unscored"), None)
        score = ScoreArtifact(status="unscored", reason="cancelled")
        return await self._close_out(
            score, failure_kind="cancelled", cancelled=True, diagnostic=reason
        )

    async def _close_out(
        self,
        score: ScoreArtifact,
        *,
        failure_kind: str | None,
        cancelled: bool,
        diagnostic: str | None = None,
    ) -> EpisodeOutcome:
        """Reconcile, clean up, record the terminal snapshot, and seal."""

        writer = self._require_writer()
        reconciliation = self._reconcile()
        self._append_receipt(
            "reconciliation",
            ReconciliationPayload(
                reconciliation_id=reconciliation.reconciliation_id,
                budget_id=reconciliation.budget_id,
                commitment_id=reconciliation.commitment_id,
                reportable=reconciliation.reportable,
                provider_cost_microusd=self._provider_cost_micros,
                environment_cost_microusd=0,
                total_cost_microusd=self._provider_cost_micros,
            ),
        )
        receipts, cleanup_result = await self._run_cleanup()
        rescue_complete: bool | None = None
        if cleanup_result.rescue_required or self._rescue_required:
            rescue_complete = await self._attempt_rescue()
        self._append_receipt(
            "cleanup",
            CleanupPayload(
                cleanup_result_id=cleanup_result.cleanup_result_id,
                cleanup_plan_id=self._require_plan().cleanup_plan_id,
                receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
                rescue_required=cleanup_result.rescue_required,
            ),
        )
        reportability = _reportability_label(
            score=score,
            rescue_required=cleanup_result.rescue_required,
            reportable=reconciliation.reportable,
            cancelled=cancelled,
        )
        comparability = "comparable"
        # ``seal`` accepts only a ``completed`` snapshot (store.py:1359); a
        # failure is carried by ``failure_kind`` plus an unscored artifact, not
        # by a ``failed`` state, which would leave the bundle unsealable.
        self._append_final_lifecycle(
            score=score,
            reconciliation_id=reconciliation.reconciliation_id,
            reportable=reconciliation.reportable,
            cleanup_result_id=cleanup_result.cleanup_result_id,
            rescue_required=cleanup_result.rescue_required,
            failure_kind=failure_kind,
        )
        outcome = self._seal(
            score=score,
            reconciliation_id=reconciliation.reconciliation_id,
            cleanup_result_id=cleanup_result.cleanup_result_id,
            reportability_label=reportability,
            comparability_label=comparability,
        )
        await self._close_session()
        # A leaked resource is a failed episode even when the task itself was
        # scored: ``lifecycle.finish_cleanup`` reaches "failed" on
        # rescue_required for the same reason, and reporting "completed" here
        # would contradict the bundle's own cleanup_incomplete label.
        failed = failure_kind is not None or cleanup_result.rescue_required
        status: EpisodeStatus = "cancelled" if cancelled else ("failed" if failed else "completed")
        return EpisodeOutcome(
            status=status,
            episode_id=self._spec.episode_id,
            bundle_root=Path(writer.root),
            score=outcome.result,
            reportability_label=reportability,
            comparability_label=comparability,
            rescue_required=cleanup_result.rescue_required or self._rescue_required,
            rescue_complete=rescue_complete,
            diagnostic=diagnostic,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reconcile(self) -> Any:
        """Close the budget with real usage, marking what could not be measured.

        A poisoned session cannot report environment usage, so those resources
        are declared unavailable. That is what makes the episode
        ``budget_unreconciled`` rather than silently reportable.
        """

        measured: dict[str, int] = {
            "provider_input_tokens": self._usage_totals.get("input_tokens", 0),
            "provider_output_tokens": self._usage_totals.get("output_tokens", 0),
            "provider_cache_tokens": (
                self._usage_totals.get("cache_read_tokens", 0)
                + self._usage_totals.get("cache_write_tokens", 0)
            ),
            "provider_usd_micros": self._provider_cost_micros,
            "cloud_usd_micros": 0,
            "instance_milliseconds": max(0, _now_ms() - self._started_ms),
            "wall_milliseconds": max(0, _now_ms() - self._started_ms),
            "model_cycles": self._model_cycles,
            "guest_actions": self._guest_actions,
            "user_simulator_turns": self._simulator_turns,
        }
        unavailable: set[str] = (
            {"cloud_usd_micros", "instance_milliseconds"} if self._rescue_required else set()
        )
        usage: list[Usage] = []
        for resource, value in measured.items():
            if resource in unavailable:
                usage.append(
                    UnavailableUsage(
                        resource=resource,  # pyright: ignore[reportArgumentType]
                        reason="worker terminated before usage could be measured",
                    )
                )
            else:
                usage.append(
                    AvailableUsage(
                        resource=resource,  # pyright: ignore[reportArgumentType]
                        value=value,
                    )
                )
        return reconcile_budget(self._spec.budget, self._spec.reservations, usage)

    async def _run_cleanup(self) -> tuple[tuple[CleanupReceipt, ...], Any]:
        """Run cleanup on the live session, or mint incomplete receipts if dead."""

        plan = self._require_plan()
        session = self._session
        receipts: tuple[CleanupReceipt, ...] = ()
        if session is not None and not self._rescue_required:
            try:
                receipts = await session.cleanup(
                    CleanupParams(
                        operation_id=f"cleanup-{self._spec.episode_id}",
                        cleanup_plan=plan,
                        action_ids=tuple(action.action_id for action in plan.actions),
                    ),
                    timeout=self._config.cleanup_timeout,
                )
            except BaseException:
                self._rescue_required = True
                receipts = ()
        if not receipts:
            # A dead worker cannot produce evidence, and "attempted" alone is
            # never evidence of cleanup, so these receipts aggregate as
            # incomplete and force rescue.
            receipts = tuple(_incomplete_receipt(plan, action.action_id) for action in plan.actions)
        return receipts, aggregate_cleanup(plan, receipts)

    async def _attempt_rescue(self) -> bool:
        descriptor = self._descriptor
        if descriptor is None:
            return False
        try:
            aggregate = await self._rescue(descriptor)
        except BaseException:
            return False
        return bool(aggregate.complete)

    def _build_manifest(self, handshake: Handshake, plan: CleanupPlan) -> EvidenceManifest:
        spec = self._spec
        return EvidenceManifest(
            episode_id=spec.episode_id,
            harness_version=spec.harness_version,
            harness_git_revision=spec.harness_git_revision,
            adapter_id=handshake.metadata.adapter_id,
            adapter_version=handshake.metadata.version,
            benchmark_id=spec.benchmark_id,
            benchmark_release=spec.benchmark_release,
            task_id=spec.task_id,
            task_digest=spec.task_digest,
            input_digest=spec.input_digest,
            requested_route=spec.requested_route,
            fallback_policy=spec.fallback_policy,
            environment_digest=spec.environment_digest,
            environment_release=spec.environment_release,
            dependency_plan_id=spec.dependency_plan.plan_id,
            budget_id=spec.budget.budget_id,
            cleanup_plan_id=plan.cleanup_plan_id,
            config_digest=spec.config_digest,
            created_wall_time_ms=self._started_ms,
            metadata=dict(spec.metadata),
        )

    def _append(self, kind: Any, payload: Any) -> None:
        writer = self._require_writer()
        try:
            writer.append(kind, payload)
        except EvidenceError as error:
            raise _EvidenceFailure(_diagnostic(error)) from error

    def _append_receipt(self, kind: str, payload: Any) -> None:
        writer = self._require_writer()
        try:
            if kind == "reconciliation":
                writer.record_reconciliation(payload)
            elif kind == "cleanup":
                writer.record_cleanup(payload)
            else:
                writer.record_scoring_result(payload)
        except EvidenceError as error:
            raise _EvidenceFailure(_diagnostic(error)) from error

    def _publish(self, data: bytes, *, media_type: str, expected_sha256: str | None = None) -> Any:
        writer = self._require_writer()
        try:
            return writer.publish_artifact(
                data, media_type=media_type, expected_sha256=expected_sha256
            )
        except EvidenceError as error:
            raise _EvidenceFailure(_diagnostic(error)) from error

    def _begin_finalization(self, intent: FinalizationIntent, operation: str | None) -> None:
        writer = self._require_writer()
        try:
            writer.begin_finalization(self._finalization_id, operation, intent)
        except EvidenceError as error:
            raise _EvidenceFailure(_diagnostic(error)) from error

    def _append_lifecycle(self, lifecycle: EpisodeLifecycle, *, state: str) -> None:
        # This is the journal's FIRST lifecycle link, so it must declare no
        # predecessor: the verifier chains lifecycle events against the events
        # it has actually seen, not against the in-process state machine, which
        # already advanced through planned/preflighted/authorized to get here.
        self._append(
            "lifecycle_transition",
            LifecycleTransitionPayload(
                previous_state_id=None,
                state_id=lifecycle.state_id,
                state=state,  # pyright: ignore[reportArgumentType]
                preflight_seal_id=lifecycle.preflight_seal_id,
                commitment_id=lifecycle.commitment_id,
            ),
        )

    def _append_final_lifecycle(
        self,
        *,
        score: ScoreArtifact,
        reconciliation_id: str,
        reportable: bool,
        cleanup_result_id: str,
        rescue_required: bool,
        failure_kind: str | None,
    ) -> None:
        writer = self._require_writer()
        lifecycle = self._lifecycle
        assert lifecycle is not None
        payload = LifecycleTransitionPayload(
            previous_state_id=lifecycle.state_id,
            state_id=canonical_digest(
                "runner-terminal-state-v1",
                {
                    "episode_id": self._spec.episode_id,
                    "finalization_id": self._finalization_id,
                    "score_id": score.score_id,
                },
            ),
            state="completed",
            finalization_id=self._finalization_id,
            preflight_seal_id=self._spec.preflight.seal_id,
            commitment_id=lifecycle.commitment_id,
            reconciliation_id=reconciliation_id,
            reconciliation_reportable=reportable,
            score_id=score.score_id,
            cleanup_result_id=cleanup_result_id,
            rescue_required=rescue_required,
            failure_kind=failure_kind,  # pyright: ignore[reportArgumentType]
        )
        try:
            writer.record_final_lifecycle(payload)
        except EvidenceError as error:
            raise _EvidenceFailure(_diagnostic(error)) from error

    def _seal(
        self,
        *,
        score: ScoreArtifact,
        reconciliation_id: str,
        cleanup_result_id: str,
        reportability_label: str,
        comparability_label: str,
    ) -> Any:
        writer = self._require_writer()
        lifecycle = self._lifecycle
        assert lifecycle is not None
        draft = OutcomeDraft(
            finalization_id=self._finalization_id,
            preflight_seal_id=self._spec.preflight.seal_id,
            commitment_id=lifecycle.commitment_id,
            reconciliation_id=reconciliation_id,
            cleanup_result_id=cleanup_result_id,
            result=score,
            reportability_label=reportability_label,  # pyright: ignore[reportArgumentType]
            comparability_label=comparability_label,  # pyright: ignore[reportArgumentType]
            ended_wall_time_ms=_now_ms(),
        )
        try:
            return writer.seal(draft)
        except EvidenceError as error:
            raise _EvidenceFailure(_diagnostic(error)) from error

    async def _abandon_after_scoring_failure(self, error: BaseException) -> EpisodeOutcome:
        writer = self._require_writer()
        rescue_complete = await self._attempt_rescue()
        await self._emergency_teardown()
        record = writer.abandon("ambiguous_finalization", _diagnostic_code(error))
        return EpisodeOutcome(
            status="abandoned",
            episode_id=self._spec.episode_id,
            bundle_root=Path(writer.root),
            rescue_required=True,
            rescue_complete=rescue_complete,
            diagnostic=record.diagnostic_code,
        )

    async def _abandon_for_evidence(self, detail: str) -> EpisodeOutcome:
        """The writer is unusable; rescue first because cloud safety needs no writer."""

        rescue_complete = await self._attempt_rescue()
        await self._emergency_teardown()
        writer = self._writer
        root = Path(writer.root) if writer is not None else None
        if writer is not None:
            try:
                writer.abandon("infrastructure_failure", "evidence-write-failed")
            except EvidenceError:
                # A writer this broken may not even be able to record its own
                # abandonment; the bundle stays unsealed and recoverable.
                pass
        return EpisodeOutcome(
            status="abandoned",
            episode_id=self._spec.episode_id,
            bundle_root=root,
            rescue_required=True,
            rescue_complete=rescue_complete,
            diagnostic=detail,
        )

    async def _close_session(self) -> None:
        session = self._session
        supervisor = self._supervisor
        if session is not None and not self._rescue_required:
            try:
                await session.close(
                    CloseParams(
                        operation_id=f"close-{self._spec.episode_id}",
                        episode_id=self._spec.episode_id,
                    ),
                    timeout=self._config.cleanup_timeout,
                )
            except BaseException:
                pass
        if supervisor is not None:
            try:
                await supervisor.terminate()
            except BaseException:
                pass

    async def _emergency_teardown(self) -> None:
        supervisor = self._supervisor
        if supervisor is not None:
            try:
                await supervisor.terminate()
            except BaseException:
                pass

    def _require_writer(self) -> EvidenceWriter:
        writer = self._writer
        assert writer is not None
        return writer

    def _require_session(self) -> VerifiedAdapterSession:
        session = self._session
        assert session is not None
        return session

    def _require_plan(self) -> CleanupPlan:
        plan = self._cleanup_plan
        assert plan is not None
        return plan


class _ProviderFailure(Exception):
    """A terminal model-provider error after the client exhausted its retries."""


def _reportability_label(
    *,
    score: ScoreArtifact,
    rescue_required: bool,
    reportable: bool,
    cancelled: bool,
) -> str:
    """Pick the single most severe label.

    Precedence is cleanup_incomplete > budget_unreconciled > unscored, because
    a leaked resource is a worse claim about the run than an unclosed budget,
    which in turn matters more than a missing score.
    """

    if rescue_required:
        return "cleanup_incomplete"
    if not reportable:
        return "budget_unreconciled"
    if cancelled:
        return "cancelled"
    if score.status != "scored":
        return "unscored"
    return "reportable"


def _terminal_kind(batch: ActionBatch) -> Literal["finish", "ask_user"] | None:
    for action in batch.actions:
        if isinstance(action, FinishAction):
            return "finish"
        if isinstance(action, AskUserAction):
            return "ask_user"
    return None


def _ask_prompt(batch: ActionBatch) -> tuple[str, str]:
    """Return the ask's protocol request ID and its question text."""

    for action in batch.actions:
        if isinstance(action, AskUserAction):
            return action.request_id, action.question
    raise ValueError("batch carries no ask-user action")


def _batch_id(batch: ActionBatch) -> str:
    return f"batch-{_adapter_batch_id(batch)[:24]}"


def _adapter_batch_id(batch: ActionBatch) -> str:
    return canonical_digest("adapter-action-batch-v1", batch)


def _incomplete_receipt(plan: CleanupPlan, action_id: str) -> CleanupReceipt:
    from local_operator.evaluation.lifecycle import record_cleanup

    return record_cleanup(
        plan,
        action_id,
        status="failed",
        evidence_code="worker-unavailable",
        duration_ms=0,
    )


def _read_artifact(root: Path, sha256: str) -> bytes:
    return (root / sha256).read_bytes()


def _diagnostic(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def _diagnostic_code(error: BaseException) -> str:
    """Derive a ``StrictIdentifier`` diagnostic code from an exception type.

    The identifier pattern forbids a leading separator, and private exception
    names here begin with an underscore, so the result is stripped and given a
    stable fallback rather than being allowed to fail validation on a path that
    is already handling a failure.
    """

    raw = "".join(char if char.isalnum() else "-" for char in type(error).__name__.lower()).strip(
        "-"
    )
    return raw[:64] or "unknown-error"


def _now_ms() -> int:
    return int(time.time_ns() // 1_000_000)

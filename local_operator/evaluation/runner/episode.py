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

from pydantic import ValidationError

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AskUserExchangeParams,
    CleanupParams,
    CloseParams,
    ExecuteParams,
    Handshake,
    InspectRequirementsParams,
    PrepareParams,
    RescueDescriptor,
    ResetStartParams,
    ResolvedSecret,
    ScopedInfraValue,
    ScoreParams,
    SecretRef,
)
from local_operator.evaluation.adapters.supervisor import (
    AdapterSupervisor,
    HostVerifier,
    VerifiedAdapterSession,
    discard_rescue,
    persist_rescue,
    run_rescue,
    verify_artifact,
)
from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    BudgetCommitmentPayload,
    CancelPayload,
    CleanupPayload,
    ContextCompactionPayload,
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
from local_operator.evaluation.runner.guards import (
    RECENT_TURNS_WINDOW,
    EpisodeGuard,
    GuardInput,
    GuardVerdict,
    default_guards,
)
from local_operator.evaluation.runner.model import EpisodeModelClient, EpisodeTurn
from local_operator.evaluation.runner.responder import NullUserResponder, UserResponder
from local_operator.evaluation.runner.secrets import (
    SecretResolver,
    StaticSecretResolver,
)

# Cleanup kinds are a closed vocabulary in ``lifecycle.CleanupAction``. A worker
# session is the one resource the parent always knows exists before the adapter
# has described anything, which is what makes the provisional plan expressible.
_PROVISIONAL_CLEANUP_ACTION = "close-session"

EpisodeStatus = Literal[
    "completed",
    "failed",
    "cancelled",
    "abandoned",
    # The writer failed AND its abandonment could not be recorded, so the bundle
    # on disk is still "open" with no terminal. This is deliberately distinct
    # from "abandoned": claiming a terminal that is not on disk is the exact
    # false report this slice exists to prevent. See _abandon_for_evidence.
    "abandonment_failed",
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

    ``guards`` are the episode guards (``runner.guards``) evaluated after every
    executed step; a firing guard truncates exactly as ``max_steps`` does, so
    the episode is still scored on the state it reached. ``None`` means
    :func:`default_guards`; an empty tuple disables them.
    ``max_cycle_cost_micros`` feeds the default cost-rate guard's absolute cap.
    ``max_decision_retries`` is how many corrective re-prompts one observation
    may take after a billed reply fails strict parsing (a ``frame_id`` the
    observation does not carry, malformed JSON) before the episode ends as a
    model failure. The default is 2 -- one re-prompt is the minimum for a
    single defect, a second covers a reply that fixes it and introduces
    another; beyond that the model is not converging and every further call
    is money spent on a batch that can never execute. ``0`` restores the old
    one-strike behaviour.
    The frame policy (``keep_recent_frames``) is deliberately NOT here: the
    model client owns its context and is built before the runner
    (``create_provider_model_client(keep_recent_frames=...)``), so a copy on
    this config would be a second declaration the runner could not enforce.
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
    max_cycle_cost_micros: int | None = None
    guards: tuple[EpisodeGuard, ...] | None = None
    max_decision_retries: int = 2


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
        secrets: SecretResolver | None = None,
        synthetic_model: bool = False,
        launch: Any = AdapterSupervisor.launch,
        rescue: Any = run_rescue,
    ) -> None:
        self._spec = spec
        self._config = config
        self._selector = selector
        self._model = model
        self._responder = responder or NullUserResponder()
        self._redactions = redactions or RedactionSet.from_resolved_values(())
        # ``None`` means "this episode has no secrets to resolve", which is only
        # true when the spec declares no refs. An empty static resolver makes a
        # spec that DOES declare refs fail as ``MissingSecret`` before any
        # allocation, rather than silently sending the worker nothing and
        # letting the adapter fail closed after it has already been launched.
        self._secrets = secrets or StaticSecretResolver({})
        self._resolved_secrets: tuple[ResolvedSecret, ...] = ()
        # Declared by the CALLER, who knows what it built: a scripted or
        # replayed model client produces a bundle that verifies exactly like
        # a real one, and nothing inside the bundle could tell them apart.
        # The label is the runner's own claim about the run, so it is set on
        # the runner rather than left to metadata a reader might not check.
        self._synthetic_model = synthetic_model
        self._launch = launch
        self._rescue = rescue

        self._writer: EvidenceWriter | None = None
        self._session: VerifiedAdapterSession | None = None
        self._supervisor: Any = None
        self._descriptor: RescueDescriptor | None = None
        self._cleanup_plan: CleanupPlan | None = None
        self._lifecycle: EpisodeLifecycle | None = None
        self._rescue_required = False
        # Every turn taken, protocol-typed: the observation the model saw and,
        # once known, the batch it chose. The model client renders these into
        # whatever conversation it sends; the runner never does.
        self._turns: list[EpisodeTurn] = []
        self._guards: tuple[EpisodeGuard, ...] = (
            default_guards(config) if config.guards is None else tuple(config.guards)
        )
        self._recent_costs: list[int] = []
        self._truncation_reason: str | None = None
        self._last_request_id: str | None = None
        self._usage_totals: dict[str, int] = {}
        self._provider_cost_micros = 0
        self._model_cycles = 0
        self._guest_actions = 0
        self._simulator_turns = 0
        self._last_exchange_id: str | None = None
        # Every route a provider actually served. A run whose route drifted from
        # the pinned one is not comparable against runs that kept it.
        self._served_routes: set[tuple[str, str, str]] = set()
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
        # _run_with_bundle records its own abandonment terminal while the
        # writer is still open, so no _EvidenceFailure escapes it here.
        return await self._run_with_bundle(handshake)

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

        # Secrets are resolved HERE -- after the handshake, before the
        # provisional descriptor and before ``prepare`` -- for two reasons.
        # First, a missing credential must fail before anything is allocated:
        # ``reset_start`` is the side-effect boundary, and a worker that has
        # to fail closed itself has already been launched and had its rescue
        # inbox entry written. Failing here surfaces as ``failed_pre_bundle``
        # with a diagnostic naming the REF, never a value. Second, every
        # resolved value must be in the redaction set before
        # ``EvidenceWriter.create`` (``_run_with_bundle``) so the writer
        # canaries every artifact and event against them from its first byte;
        # a value that is resolved after the writer opens is one the bundle
        # could carry.
        self._resolved_secrets = self._secrets.resolve([ref.name for ref in self._spec.secret_refs])
        if self._resolved_secrets:
            self._redactions = self._redactions.with_values(
                secret.value for secret in self._resolved_secrets
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
            # Tracked from the constant rather than restated: a literal here
            # silently disagrees with the models on the next protocol bump, and
            # the failure surfaces as an unrelated pre-bundle validation error.
            schema_version=ADAPTER_SCHEMA_VERSION,
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
            # The abandonment terminal MUST be recorded here, while the writer
            # is still open. Letting _EvidenceFailure propagate to run() meant
            # the finally below closed the writer first, so abandon() then hit
            # "evidence writer is closed" and the bundle was left with no
            # terminal at all -- the runner reported "abandoned" while the
            # bundle on disk stayed open forever.
            return await self._execute(handshake)
        except _EvidenceFailure as error:
            return await self._abandon_for_evidence(str(error))
        finally:
            # Still correct for FD hygiene; only its ordering against abandon()
            # was wrong. close() after a recorded terminal is a no-op for the
            # bundle's contents.
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
        # The parent owns this directory, so the parent creates it. An adapter
        # told to publish into a missing root would otherwise have to create it
        # itself, which hands the worker a say in the mode and ownership of the
        # one directory the parent later opens and reads.
        self._config.artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        result = await session.reset_start(
            ResetStartParams(
                operation_id=f"reset-{self._spec.episode_id}",
                task_id=self._spec.task_id,
                episode_id=self._spec.episode_id,
                # The worker is spawned with a stripped environment, so this RPC
                # field is its ONLY way to learn where published frame bytes are
                # read from. It is the same directory ``_record_observation``
                # verifies against, which is what keeps "where the adapter wrote"
                # and "where the parent reads" a single parent-chosen value.
                artifact_root=str(self._config.artifact_root),
                # The same values ``_launch_and_prepare`` resolved and canaried.
                # They cross only this private pipe; the descriptor on disk
                # carries the refs, and rescue re-resolves at rescue time.
                secrets=self._resolved_secrets,
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
            data = verify_artifact(self._config.artifact_root, frame.artifact)
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
        self._turns.append(EpisodeTurn(observation=observation))

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
                self._truncation_reason = self._truncation_reason or "max-steps"
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
        """Ask the model until it returns a usable batch, within the retry bound.

        A :class:`DecisionRejected` is a BILLED call whose reply failed strict
        parsing (the first paid episode's ``frame_id "1"`` against a published
        ``"screen"``). It is the model's error, not the provider's, and it is
        recoverable: the client has already folded the rejection into its
        context, so calling ``decide`` again for the same observation is a
        corrective re-prompt. Each attempt -- rejected or not -- writes its own
        request/response/usage triple and counts as a model cycle, because
        each was a real provider call; a rejected one additionally writes a
        retryable ``error`` event naming the defect, so a reader can see the
        correction happen rather than infer it from an extra triple.

        The bound is ``config.max_decision_retries`` corrective re-prompts.
        Spending it means the model is not converging, and the episode ends
        as a MODEL failure (``_ModelFailure``) rather than burning the budget
        on replies that can never execute.
        """

        rejections = 0
        while True:
            try:
                return await self._decide_once(observation)
            except _DecisionRejection as rejection:
                rejections += 1
                if rejections > self._config.max_decision_retries:
                    raise _ModelFailure(
                        f"model produced no usable decision after {rejections} attempt(s): "
                        f"{rejection.diagnostic}"
                    ) from rejection

    async def _decide_once(self, observation: Observation) -> Any:
        """One model call, writing request/response/usage in that exact order.

        The verifier binds a response to its request and a usage record to that
        response, so all three must be written even when the provider reports
        nothing -- a missing usage record leaves an unclosed operation and the
        bundle cannot reach a terminal.
        """
        from local_operator.evaluation.runner.model import DecisionRejected

        request_id = f"req-{self._model_cycles}-{uuid.uuid4().hex[:12]}"
        message_count = len(self._turns)
        # The provider is called BEFORE the request event is written, even
        # though the three events keep their required request/response/usage
        # order in the journal. A request written first would be left unclosed
        # by a provider failure, and the verifier requires every request to
        # carry its response and usage before any terminal (verify.py:760) --
        # so recording it eagerly would make the failure path unsealable.
        rejected: DecisionRejected | None = None
        try:
            decision = await self._model.decide(observation, tuple(self._turns))
        except _EvidenceFailure:
            raise
        except DecisionRejected as error:
            # A billed call with an unusable reply. Its evidence is written
            # below exactly like an accepted decision's -- the bundle's
            # counters must stay a pure sum of what was actually spent -- and
            # the rejection is then raised for ``_decide`` to bound.
            rejected = error
            decision = error
        except BaseException as error:
            from local_operator.evaluation.runner.provider_client import (
                ContextUnrecoverableError,
            )

            # The client could not fit the context into the window even after
            # pruning, summarising, and shedding stale observations. That is
            # the harness's limit, not a provider outage, so it must NOT be
            # re-classified as a provider failure: let it propagate so
            # ``_finalize_failure`` records it as a harness (adapter) error
            # and seals unscored — the honest outcome the evidence model
            # supports. A scored truncation is not representable here (the
            # last step's event was already written), and the verifier's
            # one-step-per-batch rule forbids amending it.
            if isinstance(error, ContextUnrecoverableError):
                raise
            raise _ProviderFailure(_diagnostic(error)) from error
        if decision.compaction is not None:
            # Declared BEFORE the request triple: the client rebuilt its
            # context on the way to this request, so the compaction belongs
            # between the previous request's usage receipt and this request
            # (which is where the verifier requires it). The summary the model
            # was handed is an artifact so a reader can see exactly what
            # scaffolding the harness gave it.
            record = decision.compaction
            summary_artifact = None
            if record.summary_text:
                summary_artifact = self._publish(
                    record.summary_text.encode("utf-8"), media_type="text/plain"
                )
            self._append(
                "context_compaction",
                ContextCompactionPayload(
                    compaction_id=f"compaction-{self._model_cycles}-{uuid.uuid4().hex[:12]}",
                    previous_request_id=self._last_request_id,
                    strategy=record.strategy,
                    tokens_before=record.tokens_before,
                    tokens_after=record.tokens_after,
                    frames_dropped=record.frames_dropped,
                    messages_before=record.messages_before,
                    messages_after=record.messages_after,
                    summary_artifact=summary_artifact,
                ),
            )
        # A rejected reply may carry no served route (a scripted client need
        # not know one); the requested route is the honest stand-in because
        # nothing served was accepted.
        served_route = decision.route or self._spec.requested_route
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
                prompt_cache_key=decision.prompt_cache_key,
                context_tokens=decision.context_tokens,
            ),
        )
        self._append(
            "model_response",
            ModelResponsePayload(
                request_id=request_id,
                provider_request_id=decision.provider_request_id,
                requested_route=self._spec.requested_route,
                served_route=served_route,
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
        self._last_request_id = request_id
        self._recent_costs.append(decision.cost_micros)
        self._served_routes.add(
            (
                served_route.provider_id,
                served_route.route_id,
                served_route.model_id,
            )
        )
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
        if rejected is not None:
            # The diagnostic is what the model will be shown; it is published
            # as an artifact rather than squeezed into the identifier-shaped
            # ``diagnostic_code`` so the exact correction is auditable.
            #
            # The REPLY is published alongside it, because the diagnostic
            # alone does not say what the model was attempting. A Pydantic
            # error names the fields it refused; it cannot tell a later reader
            # whether the turn was a mistyped keyboard action or a stray
            # sentence after the JSON. Without the reply, diagnosing a
            # rejection class costs a whole re-run of a paid episode.
            detail = self._publish(
                _rejection_detail(rejected).encode("utf-8"), media_type="text/plain"
            )
            self._append(
                "error",
                ErrorPayload(
                    error_id=f"err-{uuid.uuid4().hex[:12]}",
                    category="model",
                    diagnostic_code="decision-rejected",
                    detail_artifact=detail,
                    retryable=True,
                ),
            )
            raise _DecisionRejection(rejected.diagnostic) from rejected
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
        self._close_turn(batch)
        truncated = self._truncated or self._steps_taken >= self._config.max_steps
        reason = self._truncation_reason
        if truncated and reason is None:
            reason = "max-steps"
        if not truncated:
            # Guards are evaluated HERE, on the post-step snapshot and before
            # the step event is written, for the same reason ``max_steps`` is
            # folded in on this line: truncation is recorded on the last step,
            # and the verifier accepts a truncated last step as the terminal
            # (verify.py's final-step rule). A guard checked at the loop head
            # would fire one step too late, after a non-truncated step had
            # already been written, leaving the bundle without a terminal.
            verdict = self._evaluate_guards(result.observation)
            if verdict is not None:
                truncated = True
                reason = verdict.code
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
                # PROTOCOL GAP: ``ExecutionReceipt`` carries no termination
                # flag, so an adapter cannot currently say "the environment
                # ended this episode" -- only the model's own FinishAction or
                # our step cap can end the step path. If a future adapter API
                # adds that signal, it must be plumbed through here, or an
                # environment-initiated termination will be silently ignored.
                terminated=False,
                truncated=truncated,
                truncation_reason=reason if truncated else None,
            ),
        )
        self._truncated = truncated
        self._truncation_reason = reason if truncated else None
        self._last_step_terminated = truncated
        self._record_observation(result.observation)

    def _close_turn(self, batch: ActionBatch, *, ask_answer: str | None = None) -> None:
        """Attach the batch just executed to the turn it was decided on.

        The batch is unknown when the turn's observation is recorded, so the
        turn is completed here, before the next observation is appended; the
        model client relies on seeing the batch on turn ``i`` when it renders
        turn ``i+1``.
        """
        if not self._turns:
            return
        last = self._turns[-1]
        update: dict[str, Any] = {"batch": batch}
        if ask_answer is not None:
            update["ask_answer"] = ask_answer
        self._turns[-1] = last.model_copy(update=update)

    def _evaluate_guards(self, latest: Observation) -> GuardVerdict | None:
        """The first ``truncate`` verdict over the post-step snapshot, if any.

        ``latest`` is the observation the step just produced; it is included
        as an undecided turn so a frame-comparison guard sees the screen the
        last action left behind, not only the screens that were acted on.
        """
        if not self._guards:
            return None
        recent = (*self._turns[-RECENT_TURNS_WINDOW:], EpisodeTurn(observation=latest))
        snapshot = GuardInput(
            steps_taken=self._steps_taken,
            model_cycles=self._model_cycles,
            provider_cost_micros=self._provider_cost_micros,
            elapsed_ms=max(0, _now_ms() - self._started_ms),
            usage_totals=dict(self._usage_totals),
            recent_turns=recent,
            recent_costs_micros=tuple(self._recent_costs[-32:]),
            budget=self._spec.budget,
        )
        for guard in self._guards:
            verdict = guard.evaluate(snapshot)
            if verdict.kind == "truncate":
                return verdict
        return None

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
        # The answer rides on the asking turn so the model client can deliver
        # it with the observation that follows; ``_execute_batch`` then closes
        # the turn with the batch itself.
        self._close_turn(batch, ask_answer=answer)
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

        # Three failure classes, each with its own category, unscored reason
        # and lifecycle failure kind, because they answer different questions
        # for whoever reads the bundle: ``provider`` is an outage on the way to
        # the model, ``model`` is the agent under test failing to act, and
        # ``adapter`` (crash) is the harness or environment breaking.
        category: Literal["adapter", "provider", "model"]
        reason: Literal["crash", "infrastructure_failure", "model_failure"]
        failure_kind: Literal["crash", "infrastructure", "model"]
        if isinstance(error, _ProviderFailure):
            category, reason, failure_kind = "provider", "infrastructure_failure", "infrastructure"
        elif isinstance(error, _ModelFailure):
            category, reason, failure_kind = "model", "model_failure", "model"
        else:
            category, reason, failure_kind = "adapter", "crash", "crash"
        # The bounded, redaction-scanned reason for the failure, not just its
        # type. ``diagnostic_code`` is a StrictIdentifier derived from the
        # exception CLASS ("rpcremoteerror"), which is enough to bucket a
        # failure and never enough to diagnose one: the real episode
        # ep-ffda3fc88f81 burned a paid 16-step run and left a reader nothing
        # but that word and a null ``detail_artifact``. The rejected-decision
        # path above already publishes its diagnostic as an artifact for
        # exactly this reason; a fatal error deserves it at least as much,
        # because there is no retry that will produce a second chance to look.
        #
        # ``_failure_detail`` leads with that same ``_diagnostic`` line and
        # then appends the adapter's structured cause when the failure crossed
        # the RPC boundary carrying one -- without it, ep-e46c789ca818 recorded
        # the whole of "adapter_error: adapter operation failed" after 19 billed
        # steps. ``_diagnostic`` itself strips pydantic's ``input_value=`` echo
        # (the one place a resolved secret could surface) and truncates to 500
        # chars, and
        # ``publish_artifact`` independently scans every byte against the
        # episode's RedactionSet, so a leaked credential fails the write rather
        # than reaching the bundle.
        detail: Any = None
        try:
            detail = self._publish(_failure_detail(error).encode("utf-8"), media_type="text/plain")
        except (_EvidenceFailure, OSError):
            # Best-effort by design. This runs on the path that is already
            # handling a failure, and an unpublishable detail must never be the
            # reason a bundle loses its terminal -- the code, category and
            # outcome diagnostic still reach the reader without it. ``OSError``
            # is named alongside ``_EvidenceFailure`` because ``publish_artifact``
            # poisons the writer and re-raises a RAW ``OSError`` on ambiguous I/O
            # (ENOSPC/EIO), and ``_publish`` converts only ``EvidenceError`` -- so
            # a disk failure here would otherwise escape ``_finalize_failure`` and
            # cost the bundle its terminal: the exact shape this change removes.
            detail = None
        self._append(
            "error",
            ErrorPayload(
                error_id=f"err-{uuid.uuid4().hex[:12]}",
                category=category,
                diagnostic_code=_diagnostic_code(error),
                detail_artifact=detail,
                retryable=False,
            ),
        )
        self._begin_finalization(FinalizationIntent(kind="unscored"), None)
        score = ScoreArtifact(status="unscored", reason=reason)
        return await self._close_out(
            score,
            failure_kind=failure_kind,
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
        else:
            # Every declared action confirmed on the live session: this episode
            # owns nothing any more, so its descriptor leaves the rescue inbox.
            # Only this branch holds that confirmation; a rescued episode's
            # descriptor is retired by the rescue path (or the sweep) on a
            # complete aggregate, never here on hope.
            self._retire_descriptor()
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
            synthetic_model=self._synthetic_model,
        )
        comparability = _comparability_label(
            requested=self._spec.requested_route,
            served=self._served_routes,
        )
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
            # The rescue worker is a fresh process with nothing but the
            # descriptor; it needs the credentials again to tear down. These
            # are the values resolved at launch, so an in-process rescue never
            # re-reads the store mid-failure.
            aggregate = await self._rescue(descriptor, secrets=self._resolved_secrets)
        except BaseException:
            return False
        complete = bool(aggregate.complete)
        if complete:
            self._retire_descriptor()
        return complete

    def _retire_descriptor(self) -> None:
        """Unlink this episode's ``rescue.json``; only confirmed-clean callers."""

        if self._descriptor is None:
            return
        try:
            discard_rescue(self._config.rescue_root)
        except OSError:
            # A descriptor that could not be unlinked is re-rescued by the next
            # sweep (``instance-absent``) -- harmless, and strictly better than
            # failing an already-clean episode over inbox hygiene.
            pass

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
        status: EpisodeStatus = "abandoned"
        if writer is not None:
            try:
                writer.abandon("infrastructure_failure", "evidence-write-failed")
            except EvidenceError as error:
                # NEVER swallow this. `abandon()` independently re-verifies the
                # bundle and refuses on any error-severity issue, so a terminal
                # is not always recordable -- most commonly when an `append`
                # failed AFTER its artifact was already published, leaving an
                # `artifact_unreferenced` orphan that the gate rejects.
                #
                # Reporting "abandoned" over a bundle whose terminal_state is
                # still "open" is precisely the false claim this slice exists to
                # prevent, so the returned status now matches the disk and names
                # the reason. The bundle stays unsealed and recoverable, which
                # is a real state an operator can act on.
                status = "abandonment_failed"
                detail = f"{detail}; abandonment refused: {_diagnostic(error)}"
        return EpisodeOutcome(
            status=status,
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


class _DecisionRejection(Exception):
    """One billed reply failed parsing; its evidence is already written.

    Raised by ``_decide_once`` AFTER the attempt's triple and its retryable
    ``error`` event are in the journal, so ``_decide`` can count it against
    the retry bound without touching evidence itself.
    """

    def __init__(self, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.diagnostic = diagnostic


class _ModelFailure(Exception):
    """The model spent its corrective re-prompts without a usable decision.

    Distinct from ``_ProviderFailure`` because every call was answered and
    billed: the provider worked, the agent under test did not. The bundle
    seals ``category="model"`` / ``reason="model_failure"`` so a run that
    ended this way is never counted as an infrastructure outage.
    """


def _reportability_label(
    *,
    score: ScoreArtifact,
    rescue_required: bool,
    reportable: bool,
    cancelled: bool,
    synthetic_model: bool = False,
) -> str:
    """Pick the single most severe label.

    Precedence is cleanup_incomplete > budget_unreconciled > unscored, because
    a leaked resource is a worse claim about the run than an unclosed budget,
    which in turn matters more than a missing score. ``synthetic_model`` sits
    last: it only decides whether a run that would otherwise be reportable
    is, because a scripted model's bundle is otherwise indistinguishable from
    a real result, and every more severe label already says "not a result".
    """

    if rescue_required:
        return "cleanup_incomplete"
    if not reportable:
        return "budget_unreconciled"
    if cancelled:
        return "cancelled"
    if score.status != "scored":
        return "unscored"
    if synthetic_model:
        return "synthetic_model"
    return "reportable"


def _comparability_label(
    *,
    requested: RouteIdentity,
    served: set[tuple[str, str, str]],
) -> str:
    """Report a route change, which is the whole reason the served route is carried.

    A silent provider fallback is exactly what invalidates a comparison against
    runs that held the pinned route, so a run that drifted must not seal as
    ``comparable`` however well it scored. A run that made no model call has
    nothing to contradict its pin and stays comparable.
    """

    pinned = (requested.provider_id, requested.route_id, requested.model_id)
    if served - {pinned}:
        return "route_changed"
    return "comparable"


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


def _failure_detail(error: BaseException) -> str:
    """The fatal-error artifact: the diagnostic, plus the adapter's own cause.

    ``_diagnostic`` is also the ``outcome.diagnostic`` field and is bounded at
    500 characters for that reason. An adapter's structured cause -- type,
    message, method, operation ID, cause chain and worker-side frames -- is
    legitimately longer than that, and truncating it here would reintroduce
    exactly the loss this artifact exists to prevent, one layer further out.
    So the artifact carries both: the same first line a reader sees in the
    outcome, then the full structured detail when the failure crossed the
    adapter boundary carrying one.

    Every field rendered below was bounded and canary-checked on the WORKER
    side before it crossed (worker._error_detail), and ``publish_artifact``
    independently scans these bytes against the episode's own RedactionSet, so
    a leak fails the write rather than reaching the bundle.
    """

    summary = _diagnostic(error)
    detail = getattr(error, "detail", None)
    if detail is None or not hasattr(detail, "render"):
        return summary
    lines = [
        summary,
        "",
        "--- adapter detail ---",
        f"exception_type: {detail.exception_type}",
        f"message: {detail.message}",
        f"method: {detail.method}",
    ]
    if detail.operation_id is not None:
        lines.append(f"operation_id: {detail.operation_id}")
    for index, cause in enumerate(detail.causes, start=1):
        lines.append(f"cause[{index}]: {cause.exception_type}: {cause.message}")
    for frame in detail.frames:
        lines.append(f"  at {frame.file}:{frame.line} in {frame.function}")
    return "\n".join(lines)


def _rejection_detail(rejected: Any) -> str:
    """The rejection artifact: why the reply was refused AND what it said.

    Two sections rather than one blob, so a reader (or a script mining a batch
    of bundles for rejection classes) can tell the harness's diagnostic apart
    from the model's own words. A client that did not capture the reply --
    every implementation of the protocol is free not to -- degrades to the
    diagnostic alone rather than emitting an empty section that reads as "the
    model said nothing".
    """

    reply = getattr(rejected, "reply", None)
    if not reply:
        return rejected.diagnostic
    return f"{rejected.diagnostic}\n\n--- rejected reply ---\n{reply}"


def _diagnostic(error: BaseException) -> str:
    """Render an error for ``EpisodeOutcome.diagnostic`` without its inputs.

    A pydantic ``ValidationError``'s ``str()`` embeds ``input_value=<head>…<tail>``
    for every failing field. The one model on this boundary that carries
    secret bytes is ``ResolvedSecret``, and a value the wire bound refuses
    (a PEM-shaped credential past ``max_length``) would otherwise be quoted
    straight into the outcome JSON that a paid run redirects into its durable
    root. The resolvers already translate that case to a name-only error;
    this is the second line, for ANY validation error on any path: only the
    model title, the field location and the error type are kept, never the
    input. The location itself is safe -- it is a field NAME (``secrets``,
    ``1``, ``value``), not the secret's name or bytes.
    """

    if isinstance(error, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['type']}"
            for item in error.errors(include_input=False, include_url=False)
        )
        return f"ValidationError: {error.title} ({details})"[:500]
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

"""OSWorldV2Adapter: the ten-method EvaluationAdapter surface.

State the adapter holds per episode: the ``TaskDescriptor`` (from a Tier-1
static parse), the ``ProvisioningPlan`` and ``CleanupRefs`` (minted in
``prepare``), the infra values the host supplied, and — after ``reset_start``
— the live ``EnvironmentProvider`` and the cached current observation.

Method contract notes (each is the load-bearing decision a future reader would
otherwise "simplify" away):

- ``prepare`` creates NOTHING. The runner's two-stage rescue persistence
  (episode.py:289-292) depends on it: a resource allocated in ``prepare``
  exists before the descriptor naming it is durable, and leaks. So
  ``prepare`` resolves a pure plan and returns the cleanup plan; the provider
  is not even constructed here.
- ``reset_start`` is the side-effect boundary. It injects infra env values,
  constructs the provider, allocates, and captures observation 0. The
  descriptor naming the instance tag is already durable by now.
- ``score`` returns a SCORED artifact or raises. Never ``unscored``, never 0.0
  for "could not evaluate".
- ``cleanup`` reports ``succeeded`` ONLY after the provider confirms the
  terminal state; an unconfirmed terminate is ``attempted``, which
  ``aggregate_cleanup`` correctly treats as rescue-required.
- ``begin_rescue`` rebuilds refs from the descriptor's resource_refs alone —
  the worker has never run prepare.

INFEASIBLE-TASK EXCLUSION: the runner returns on a ``finish`` batch WITHOUT
calling ``execute`` (episode.py:531-534), so the adapter never sees the
terminal action and cannot push ``DONE``/``FAIL`` into OSWorld's
``action_history``, which ``evaluate()`` reads to score ``infeasible`` tasks.
An agent that correctly declares such a task infeasible would score 0. PR 1
EXCLUDES infeasible tasks rather than fabricate a FAIL the agent never sent —
that would be score fraud. See README.md "Known scope limitations".
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import time
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, Callable

from lop_osworld_v2_adapter import actions
from lop_osworld_v2_adapter import cleanup as cleanup_mod
from lop_osworld_v2_adapter import provisioning
from lop_osworld_v2_adapter import requirements as requirements_mod
from lop_osworld_v2_adapter import scoring, taskfile, vendor_bridge
from lop_osworld_v2_adapter.observation import ObservationBuilder
from lop_osworld_v2_adapter.providers.base import EnvironmentProvider

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AckResult,
    AdapterCapabilities,
    AdapterMetadata,
    AskUserExchangeParams,
    AskUserExchangeResult,
    BeginRescueParams,
    CleanupOutcome,
    CleanupParams,
    CleanupResult,
    CloseParams,
    ExecuteParams,
    ExecuteResult,
    ExecutionReceipt,
    InspectRequirementsParams,
    ObservationResult,
    ObserveParams,
    PrepareParams,
    PrepareResult,
    RequirementsResult,
    ResetStartParams,
    ScopedInfraValue,
    ScoreParams,
    ScoreResult,
)
from local_operator.evaluation.adapters.discovery import (
    AdapterDiscoveryError,
    distribution_digest,
)

_DISTRIBUTION = "lop-osworld-v2-adapter"
_ADAPTER_ID = "osworld-v2"
_VERSION = "0.1.0"
_ENTRY_POINT = "lop_osworld_v2_adapter:create"


class AdapterStateError(RuntimeError):
    """A method was called out of order or without the state it needs."""


class InfeasibleTaskExcluded(RuntimeError):
    """The task grades a refusal this adapter cannot report honestly.

    Raised from ``reset_start`` so the episode fails BEFORE any resource is
    allocated. See ``TaskDescriptor.is_infeasible``: the runner never hands the
    adapter the terminal action, so OSWorld's ``action_history`` never receives
    ``FAIL``, and a correct refusal would grade 0. Refusing the run is honest;
    fabricating the ``FAIL`` would not be.
    """


def _terminate_status(code: str) -> str:
    """Map a teardown evidence code to a cleanup status that cannot over-claim.

    This is the orphaned-instance safety net, and the mapping is asymmetric on
    purpose. ``aggregate_cleanup`` sets ``rescue_required`` only for
    ``attempted``/``failed``; it treats ``not_needed`` as CLEAN. So the two
    statuses that retire the rescue obligation must be reachable ONLY by
    positive evidence:

    * ``succeeded`` -- ``terminate_instances`` returned AND a follow-up
      ``describe_instances`` confirmed ``shutting-down``/``terminated``. A 200
      from the terminate call alone is not proof.
    * ``not_needed`` -- the tag matched no instance, i.e. we looked and there
      was nothing there.

    Everything else, including any code this build does not recognise, is
    ``attempted``: cleanup was reached but not confirmed, which correctly
    leaves ``rescue_required`` set. Defaulting the unknown case to ``attempted``
    rather than ``not_needed`` means a future provider that invents a new code
    fails safe -- toward a redundant rescue, never toward a silent leak.
    """

    if code == cleanup_mod.EVIDENCE_INSTANCE_TERMINATED:
        return "succeeded"
    if code == cleanup_mod.EVIDENCE_INSTANCE_ABSENT:
        return "not_needed"
    return "attempted"


# A provider factory takes no argument and returns a backend. Injectable so
# PR 1's tests (and the whole cloud-free slice) drive the real adapter against
# FakeProvider; PR 2 installs the AWS factory for production.
ProviderFactory = Callable[[], EnvironmentProvider]


class OSWorldV2Adapter:
    """Satisfies EvaluationAdapter. ``metadata`` is built at construction."""

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        # ``provider_factory`` is injectable so tests drive the real adapter
        # against FakeProvider in-process. When None — the production path —
        # PR 2's AWS factory is used; PR 1 raises if reached without one.
        self._provider_factory = provider_factory
        # The workspace root is where task modules live (``tasks/<id>.py``).
        # Defaulted to the worker's cwd, which the supervisor sets to the
        # workspace (supervisor.py:270).
        self._workspace_root = workspace_root or Path(os.getcwd())
        self.metadata = AdapterMetadata(
            adapter_id=_ADAPTER_ID,
            distribution=_DISTRIBUTION,
            version=_VERSION,
            entry_point=_ENTRY_POINT,
            package_digest=self._package_digest(),
            release_digest=self._release_digest(),
            # Tracked from the harness constant, never restated as a literal: a
            # literal here would keep validating against an older wire shape
            # after a protocol bump, and the handshake's exact-pin check would
            # then fail as an opaque selector mismatch rather than naming the
            # version drift.
            schema_version=ADAPTER_SCHEMA_VERSION,
            capabilities=AdapterCapabilities(
                routes=("computer",),
                ask_user=True,  # honest ONLY because V2 has user_simulator
                scoring=True,
            ),
        )
        # Per-episode state. None until the corresponding method runs.
        self._task: taskfile.TaskDescriptor | None = None
        self._plan: provisioning.ProvisioningPlan | None = None
        self._refs: cleanup_mod.CleanupRefs | None = None
        self._infra_values: tuple[ScopedInfraValue, ...] = ()
        self._provider: EnvironmentProvider | None = None
        self._observation_builder: ObservationBuilder | None = None
        self._current_observation: Any = None
        self._sequence = 0

    def _read_provider_config(self) -> dict[str, Any]:
        """Read the adapter-owned provider selection from the workspace.

        The workspace is adapter-owned, digest-pinned input, and the worker's
        cwd (supervisor.py:270). An OPTIONAL ``adapter-provider.json`` there
        names which provider ``reset_start`` builds, so a cloud-free build
        (PR 1, and every CI/test run) selects the fake WITHOUT touching the
        stripped environment. Absent the file, the production default is the
        AWS provider (PR 2); PR 1 raises rather than guess. The worker env is
        deliberately NOT used for this: it is locale/temp only
        (supervisor._ENV_ALLOW), and a selection that rode in as an env var
        would be invisible to the workspace digest that attests the run.
        """

        path = self._workspace_root / "adapter-provider.json"
        try:
            payload = json.loads(path.read_bytes())
        except OSError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _package_digest() -> str:
        """The digest of the installed wheel's RECORD, when installed.

        ``create()`` (the production entry point) always runs against the
        installed wheel, so the digest is real there; ``distribution()``
        succeeds and ``_record_rows`` verifies it. In-process unit tests load
        the source tree, where the distribution either is absent
        (``PackageNotFoundError``) or resolves to a dist-info whose RECORD is
        unreachable from the source path — both ``AdapterDiscoveryError``
        subclasses of RuntimeError. Those tests compute the real digest from
        the separately-built wheel (see the test conftest for the
        source-vs-wheel split). Only those two "not installed" signals fall
        back to the zero digest; any other error is a real integrity failure
        and propagates.
        """

        try:
            return distribution_digest(distribution(_DISTRIBUTION))
        except importlib.metadata.PackageNotFoundError:
            return "0" * 64
        except AdapterDiscoveryError:
            return "0" * 64

    @staticmethod
    def _release_digest() -> str:
        """Read the release digest the workspace's adapter-release.json pins.

        The workspace manifest carries exactly ``{"release_digest": ...}``;
        the adapter repeats it in its metadata so the handshake's exact-pin
        check binds the running code to the same attestation the workspace
        carries. The worker's cwd IS the workspace (supervisor spawns with
        ``cwd=selector.workspace``). Absent the file (in-process tests) a zero
        digest is used; the handshake pin is then the test's own fixture.
        """

        manifest = Path(os.getcwd()) / "adapter-release.json"
        try:
            payload = json.loads(manifest.read_bytes())
        except OSError:
            return "0" * 64
        digest = payload.get("release_digest", "")
        if isinstance(digest, str) and len(digest) == 64:
            return digest
        return "0" * 64

    # ------------------------------------------------------------------
    # inspect_requirements: Tier-1 static parse, no import, no network
    # ------------------------------------------------------------------

    async def inspect_requirements(self, params: InspectRequirementsParams) -> RequirementsResult:
        # Requirements must be knowable before prepare, but the task is named
        # by the runner only at reset_start. When no task is loaded yet we
        # emit the unconditional baseline so preflight can proceed; prepare
        # re-derives the full set from the parsed task. This is honest: the
        # always-on requirements are unconditional, and every conditional one
        # is re-derived from the descriptor the moment it exists.
        if self._task is None:
            baseline = taskfile.TaskDescriptor(
                task_id="unknown",
                instruction="",
                source_sha256="0" * 64,
            )
            return RequirementsResult(requirements=requirements_mod.derive_requirements(baseline))
        return RequirementsResult(requirements=requirements_mod.derive_requirements(self._task))

    # ------------------------------------------------------------------
    # prepare: declarative, allocates nothing
    # ------------------------------------------------------------------

    async def prepare(self, params: PrepareParams) -> PrepareResult:
        # ADAPTER CONTRACT (episode.py:289-292): prepare must not create
        # environment resources. This method performs no cloud API call and
        # does not construct the provider. It stores the infra values, resolves
        # a pure plan (when the task is already known), mints the deterministic
        # cleanup refs, and returns the plan the parent persists BEFORE any
        # side effect exists.
        self._refs = cleanup_mod.CleanupRefs.mint(params.episode_id)
        self._infra_values = params.infra_values
        vendor_bridge.inject_infra_environment(params.infra_values)
        if self._task is not None:
            self._plan = provisioning.resolve(
                self._task,
                episode_id=params.episode_id,
                infra_values=params.infra_values,
            )
        return PrepareResult(
            cleanup_plan=cleanup_mod.build_cleanup_plan(params.episode_id, self._refs)
        )

    # ------------------------------------------------------------------
    # reset_start: the side-effect boundary
    # ------------------------------------------------------------------

    async def reset_start(self, params: ResetStartParams) -> AckResult:
        if self._refs is None:
            raise AdapterStateError("reset_start before prepare")
        # Tier 1 static parse populates the descriptor; Tier 2 (the live
        # import) is deferred to scoring and only if OSWorld's machinery needs
        # it. Either way it happens AFTER the cleanup plan is durable.
        if self._task is None:
            source = self._load_task_source(params.task_id)
            self._task = taskfile.load_static(source, module_name=self._task_path(params.task_id))
        if self._task.is_infeasible():
            # Refused BEFORE anything is allocated, so the exclusion costs
            # nothing and cannot half-run. Documentation alone would not stop
            # an operator-built workspace from including such a task and
            # scoring an agent's correct refusal as a failure; see
            # TaskDescriptor.is_infeasible for why we will not fake the FAIL.
            raise InfeasibleTaskExcluded(
                f"task {params.task_id!r} grades a refusal via evaluator "
                "func='infeasible', which this adapter cannot score honestly: "
                "the runner never delivers the terminal action, so OSWorld's "
                "action_history never receives FAIL"
            )
        self._plan = provisioning.resolve(
            self._task,
            episode_id=params.episode_id,
            infra_values=self._infra_values,
        )

        provider = self._build_provider()
        await provider.allocate(self._plan, self._task)
        self._provider = provider

        # Capture observation 0 eagerly so the runner's immediately-following
        # observe is free and the sequence counter starts from a known state.
        #
        # The root arrives as a validated RPC field (schema 1.1), which is the
        # worker's ONLY way to learn it: the supervisor builds the child
        # environment from a closed allowlist (locale and temp), so nothing
        # ambient carries it. The parent creates the directory and refuses a
        # reset whose root differs from the one its verifier reads, so writing
        # exactly here is what makes a frame verifiable rather than a guess.
        self._observation_builder = ObservationBuilder(Path(params.artifact_root))
        raw = await provider.observe()
        self._sequence = 0
        self._current_observation = self._observation_builder.build(
            raw,
            task_id=params.task_id,
            episode_id=params.episode_id,
            sequence=0,
        )
        return AckResult()

    def _load_task_source(self, task_id: str) -> bytes:
        path = Path(self._task_path(task_id))
        try:
            return path.read_bytes()
        except OSError as error:
            raise AdapterStateError(f"task module {task_id!r} is not in the workspace") from error

    def _task_path(self, task_id: str) -> str:
        return str(self._workspace_root / "tasks" / f"{task_id}.py")

    def _build_provider(self) -> EnvironmentProvider:
        # Precedence: an injected factory (in-process tests) wins; then the
        # workspace's adapter-provider.json selects the backend; then the
        # production default (AWS, PR 2). The fake branch is reachable ONLY
        # when the workspace declares it, so a production run can never
        # silently fall back to a fake.
        if self._provider_factory is not None:
            return self._provider_factory()
        config = self._read_provider_config()
        kind = config.get("provider")
        if kind == "fake":
            from lop_osworld_v2_adapter.providers.fake import FakeProvider

            return FakeProvider(
                scripted_score=float(config.get("scripted_score", 1.0)),
                has_user_simulator=bool(config.get("has_user_simulator", False)),
            )
        if kind in (None, "aws"):
            raise AdapterStateError(
                "the AWS provider is PR 2; PR 1 runs only an injected or "
                "workspace-declared fake provider"
            )
        raise AdapterStateError(f"unknown provider selection {kind!r}")

    # ------------------------------------------------------------------
    # observe / execute
    # ------------------------------------------------------------------

    async def observe(self, params: ObserveParams) -> ObservationResult:
        if self._current_observation is None:
            raise AdapterStateError("observe before reset_start")
        return ObservationResult(observation=self._current_observation)

    async def execute(self, params: ExecuteParams) -> ExecuteResult:
        if self._provider is None or self._observation_builder is None:
            raise AdapterStateError("execute before reset_start")
        current = self._current_observation
        assert current is not None

        # Compile against the CURRENT observation's frame geometry, so the
        # coordinate space is the one the model actually saw. The builder only
        # ever produces an observation with the screen frame, so a frameless
        # current observation is unreachable and asserted, not defaulted.
        assert current.frames, "current observation carries no screen frame"
        geometry = current.frames[0].geometry
        statements = actions.compile_batch(params.action_batch, geometry)
        guest_lines = [s for s in statements if not s.startswith("WAIT ")]
        waits = [int(s.split(" ", 1)[1]) for s in statements if s.startswith("WAIT ")]
        if guest_lines or waits:
            if guest_lines:
                await self._provider.execute(guest_lines)
            for wait_ms in waits:
                await asyncio.sleep(wait_ms / 1000.0)
            if not guest_lines and waits:
                # A pure-wait batch still advances the environment's clock.
                await self._provider.execute([])

        raw = await self._provider.observe()
        self._sequence += 1
        output = self._observation_builder.build(
            raw,
            task_id=params.action_batch.task_id,
            episode_id=params.action_batch.episode_id,
            sequence=self._sequence,
        )
        receipt = ExecutionReceipt(
            operation_id=params.operation_id,
            action_batch_id=params.action_batch_id,
            input_observation_id=current.observation_id,
            output_observation_id=output.observation_id,
            sequence=output.sequence,
        )
        self._current_observation = output
        return ExecuteResult(observation=output, receipt=receipt)

    # ------------------------------------------------------------------
    # ask_user_exchange: refuse when the task has no simulator
    # ------------------------------------------------------------------

    async def ask_user_exchange(self, params: AskUserExchangeParams) -> AskUserExchangeResult:
        # Two-phase: answer is None = begin, set = finish. When the task
        # declares no user_simulator, refusing is the honest, recordable
        # answer — inventing one would be a benchmark violation. Detection is
        # from the parsed task descriptor, never from probing the provider.
        has_simulator = isinstance(self._task.user_simulator, dict) if self._task else False
        if params.answer is None:
            return AskUserExchangeResult(ask_id=params.ask_id, accepted=has_simulator)
        # Finish phase: the harness's responder already produced the answer
        # the model sees; we notify the benchmark's simulator for the record.
        if has_simulator and self._provider is not None:
            await self._provider.respond(params.prompt)
        return AskUserExchangeResult(ask_id=params.ask_id, accepted=has_simulator)

    # ------------------------------------------------------------------
    # score: scored or raise
    # ------------------------------------------------------------------

    async def score(self, params: ScoreParams) -> ScoreResult:
        if self._provider is None or self._task is None:
            raise scoring.ScoringUnavailable("score before reset_start")
        if not self._task.has_evaluator():
            # NOT 0.0: a task with no evaluator scored as failed would record
            # a failure the agent did not commit.
            raise scoring.ScoringUnavailable("task declares no evaluator")
        raw = await self._provider.evaluate()
        return ScoreResult(score=scoring.score_to_artifact(raw))

    # ------------------------------------------------------------------
    # cleanup / close / begin_rescue
    # ------------------------------------------------------------------

    async def cleanup(self, params: CleanupParams) -> CleanupResult:
        outcomes: list[CleanupOutcome] = []
        for action_id in params.action_ids:
            action = next(
                (a for a in params.cleanup_plan.actions if a.action_id == action_id), None
            )
            assert action is not None
            status, code, duration = await self._run_cleanup_action(
                action.kind, action.resource_ref
            )
            outcomes.append(
                CleanupOutcome(
                    action_id=action_id,
                    status=status,  # type: ignore[arg-type]
                    evidence_code=code,
                    duration_ms=duration,
                )
            )
        return CleanupResult(outcomes=tuple(outcomes))

    async def _run_cleanup_action(self, kind: str, resource_ref: str) -> tuple[str, str, int]:
        start = time.monotonic()

        def elapsed() -> int:
            return int((time.monotonic() - start) * 1000)

        provider = self._provider
        if kind == "release_instance":
            if provider is None:
                # NOT not_needed. "not_needed" asserts the instance is gone;
                # having no provider means we could not LOOK. aggregate_cleanup
                # treats not_needed as clean and clears rescue_required, so
                # claiming it here would retire the rescue obligation for a
                # resource nobody checked -- an orphaned instance billing
                # forever. "attempted" is the honest report: cleanup was
                # reached, cleanup was not confirmed, so a human or a rescue
                # worker must still look.
                return "attempted", cleanup_mod.EVIDENCE_TERMINATE_UNCONFIRMED, elapsed()
            code = await provider.terminate(resource_ref)
            return _terminate_status(code), code, elapsed()
        if kind == "revoke_lease":
            if provider is None:
                return "attempted", cleanup_mod.EVIDENCE_SCHEDULE_ABSENT, elapsed()
            code = await provider.delete_schedule(resource_ref)
            # A TTL schedule that is genuinely absent needs no deletion, and
            # unlike an instance its absence is directly observed rather than
            # assumed, so not_needed is the accurate status here.
            if code == cleanup_mod.EVIDENCE_SCHEDULE_DELETED:
                return "succeeded", code, elapsed()
            if code == cleanup_mod.EVIDENCE_SCHEDULE_ABSENT:
                return "not_needed", code, elapsed()
            return "attempted", code, elapsed()
        if kind == "close_session":
            # The provider reference must SURVIVE cleanup: the canonical plan
            # orders close-session before release-instance, and terminating the
            # instance needs the provider. Dropping it here would make
            # release-instance report not_needed and leak the resource the
            # cleanup plan exists to destroy. ``close`` (the protocol method)
            # is what actually releases the session afterwards. Never raises:
            # a dead env is still a closed session.
            self._current_observation = None
            return "succeeded", cleanup_mod.EVIDENCE_SESSION_CLOSED, elapsed()
        # An action KIND this build does not handle. Unreachable while
        # build_cleanup_plan emits only the three canonical kinds, but
        # CleanupActionKind is a six-member Literal, so PR 2 adding
        # delete_volume or restore_snapshot makes it reachable -- and the plan
        # arrives from a PERSISTED descriptor during rescue, which may have
        # been authored by a different adapter build than the one executing it.
        #
        # Same rule as the unknown-code case: only positive evidence of
        # teardown may retire the rescue obligation. Returning not_needed here
        # would assert "there was nothing to do" about a resource this build
        # cannot even name, clearing rescue_required for something nobody
        # released. "attempted" fails safe toward a redundant rescue.
        return "attempted", cleanup_mod.EVIDENCE_KIND_UNSUPPORTED, elapsed()

    async def begin_rescue(self, params: BeginRescueParams) -> AckResult:
        # Rebuild refs from the descriptor's plan alone. The rescue worker has
        # never run prepare; this is the proof the refs are self-describing.
        self._refs = cleanup_mod.CleanupRefs.from_descriptor_actions(
            params.descriptor.cleanup_plan.actions
        )
        # A rescue worker still needs a provider to terminate the instance.
        # PR 1's in-process rescue uses the injected factory; PR 2 builds the
        # AWS provider from the descriptor's infra_values.
        if self._provider is None and self._provider_factory is not None:
            self._provider = self._provider_factory()
        return AckResult()

    async def close(self, params: CloseParams) -> AckResult:
        # Must not raise even if the env is already dead.
        self._provider = None
        self._current_observation = None
        return AckResult()

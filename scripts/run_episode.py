#!/usr/bin/env python3
"""Run ONE evaluation episode end to end and print its outcome as JSON.

This is an OPERATOR SCRIPT, not a CLI or TUI surface: it exists so the paid
OSWorld proof and the pilot can drive a real episode -- real spawned worker,
real provider-backed model client, real credential store, real sealed bundle
-- from one command line with every input pinned on it. It deliberately
imports nothing from ``local_operator.session`` or the TUI; the runner's
isolation rule (``runner/test_isolation.py``) is what makes a bundle
reproducible from its inputs, and this script must not undo that by pulling
the operator's live session configuration into the episode.

Inputs, all explicit:

* ``--selector <json>`` -- the ``AdapterSelector`` (exact wheel, interpreter,
  workspace and digests). Computed per the adapter README, never guessed.
* ``--task-id`` -- one task in the workspace's ``tasks/``.
* ``--route provider/model`` -- the model route the episode is pinned to.
  ``fallback_policy`` is ``forbid``: a provider outage fails the episode as a
  provider error rather than silently serving a different model.
* ``--run-root`` -- ONE durable directory under which ``evidence/``,
  ``artifacts/`` and ``rescue/`` are created. It MUST NOT be under ``/tmp``
  (``runner.durable_root``): a purge of ``/private/tmp`` mid-run destroyed a
  previous paid pilot's outputs and left an instance running with no
  descriptor naming it.
* ``--secret-env NAME`` (repeatable) -- the names of environment variables
  holding secrets the task needs. These are read from THIS process's
  environment by an explicit ``EnvSecretResolver``; a name not listed here is
  never reachable, however it is spelled. Anything not listed is resolved
  from the credential store (``CredentialStoreResolver``), the same store the
  model client is built from.
* ``--infra NAME=VALUE`` (repeatable) -- non-secret infra values
  (``AWS_REGION``, ``AWS_SUBNET_ID``, ...) with ``--infra-purpose``.
* Budget and step caps (``--max-steps``, ``--max-usd``, ``--max-wall-s``,
  ``--max-cycle-usd``), all explicit; the defaults are conservative because
  a paid episode with no cap is the one thing this script must not run.

Output: the ``EpisodeOutcome`` as a JSON object on stdout, plus
``bundle_root``. Exit status is 0 only for ``completed``; ``failed``,
``cancelled``, ``abandoned``, ``abandonment_failed`` and ``failed_pre_bundle``
exit 1, and a missing secret (which fails BEFORE any allocation) exits 2 with
the secret's NAME -- never a value -- on stderr.

The exact command the paid proof runs is in the adapter README under
"Running one episode".
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from local_operator.evaluation.adapters.api import (
    AdapterSelector,
    ResolvedSecret,
    ScopedInfraValue,
    SecretRef,
)
from local_operator.evaluation.adapters.supervisor import AdapterSupervisor
from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.receipts import (
    BUDGET_RESOURCES,
    BudgetAuthorization,
    CappedAllowance,
    ComputeRequirement,
    DependencyPlan,
    RedactionSet,
    ResourceAmount,
    UncappedAllowance,
    record_preflight,
    reserve_budget,
    seal_preflight,
)
from local_operator.evaluation.runner.durable_root import (
    VolatileRootError,
    refuse_volatile_root,
)
from local_operator.evaluation.runner.episode import (
    EpisodeConfig,
    EpisodeOutcome,
    EpisodeRunner,
    EpisodeSpec,
)
from local_operator.evaluation.runner.route_ids import fold_model_id
from local_operator.evaluation.runner.secrets import (
    EnvSecretResolver,
    MissingSecret,
    SecretResolver,
)

EXIT_OK = 0
EXIT_EPISODE = 1
EXIT_PREFLIGHT = 2

# Names the OSWorld adapter (and any adapter following its contract) declares
# as secrets when it is asked ``inspect_requirements``. The runner resolves
# exactly ``spec.secret_refs``; this script needs to KNOW the refs before it
# builds the spec, and the only honest way is to ask for them on the command
# line (``--secret``). The default covers the AWS provider so the paid proof's
# command line stays short.
_DEFAULT_SECRET_REFS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


class _LayeredResolver:
    """Env-listed names from the process environment, everything else from the store.

    Two resolvers rather than one mapping so the boundary is visible: a name
    is EITHER on ``--secret-env`` (and read from this process) OR resolved
    from the credential store. Nothing here reads ``os.environ`` for a name
    the operator did not list.
    """

    def __init__(self, env_names: Sequence[str], store: SecretResolver | None) -> None:
        self._env_names = frozenset(env_names)
        self._env = EnvSecretResolver({name: os.environ.get(name, "") for name in env_names})
        self._store = store

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        resolved: list[ResolvedSecret] = []
        for name in names:
            if name in self._env_names:
                resolved.extend(self._env.resolve([name]))
            elif self._store is not None:
                resolved.extend(self._store.resolve([name]))
            else:
                raise MissingSecret(name)
        return tuple(resolved)


def _parse_route(value: str) -> tuple[str, str]:
    provider, sep, model = value.partition("/")
    if not sep or not provider or not model:
        raise argparse.ArgumentTypeError("--route must be <provider>/<model-id>")
    return provider, model


def _route_identity(provider: str, model: str) -> RouteIdentity:
    """The sealed route. ``model_id`` is the LOSSLESS fold of the provider's id.

    ``StrictIdentifier`` forbids ``/``, which every OpenRouter id carries.
    ``route_ids.fold_model_id`` is reversible (``unfold_model_id``), so the
    sealed identity still means exactly one model -- comparability depends
    on that -- and ``run`` also records the raw id in the manifest metadata
    (``route_model_id``) so nobody has to decode by hand.
    """

    folded = fold_model_id(model)
    return RouteIdentity(provider_id=provider, route_id=f"{provider}:{folded}", model_id=folded)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _harness_git_revision() -> str:
    """The installed harness's git revision, or a digest of its version.

    A source checkout answers ``git rev-parse``; a wheel install does not,
    and the manifest still needs a 64-hex value, so the version string is
    hashed as a stable stand-in and the real version rides in ``metadata``.
    """

    repo = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        completed = None
    if completed is not None and completed.returncode == 0:
        head = completed.stdout.strip()
        if len(head) == 40:
            # A 40-hex SHA-1 is not the 64-hex the manifest wants; hash it so
            # the field is stable and the raw commit lands in metadata.
            return _digest(head)
    return _digest(_harness_version())


def _harness_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("local-operator")
    except PackageNotFoundError:
        return "0.0.0"


def _task_digest(selector: AdapterSelector, task_id: str) -> str:
    """The task's own bytes, hashed: what the manifest pins as ``task_digest``."""

    path = Path(selector.workspace) / "tasks" / f"{task_id}.py"
    if not path.is_file():
        raise SystemExit(f"task {task_id!r} is not in the workspace: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _budget(
    episode_id: str,
    *,
    max_usd_micros: int,
    max_wall_ms: int,
    max_steps: int,
) -> BudgetAuthorization:
    """Cap what the operator asked to cap; leave the rest uncapped but reported.

    Every resource must appear exactly once. Token counts and cloud cost are
    reported rather than capped (there is no honest number to put on them up
    front), but the USD, wall and action caps are hard because they are the
    three things that bound a paid episode's damage.
    """

    now_ms = int(__import__("time").time() * 1000)
    capped: dict[str, int] = {
        "provider_usd_micros": max_usd_micros,
        "wall_milliseconds": max_wall_ms,
        "guest_actions": max_steps * 8,
        "model_cycles": max_steps * 2,
    }
    allowances: list[Any] = []
    for resource in BUDGET_RESOURCES:
        if resource in capped:
            allowances.append(
                CappedAllowance(resource=resource, value=capped[resource], reporting="required")
            )
        else:
            allowances.append(
                UncappedAllowance(
                    resource=resource,
                    reason="reported, not capped, by scripts/run_episode.py",
                    authorized_by="run_episode",
                    authorized_at_ms=now_ms,
                    reporting="required",
                )
            )
    return BudgetAuthorization(episode_id=episode_id, allowances=tuple(allowances))


def build_spec(
    *,
    episode_id: str,
    selector: AdapterSelector,
    task_id: str,
    route: RouteIdentity,
    benchmark_id: str,
    benchmark_release: str,
    secret_refs: Sequence[str],
    infra_values: Sequence[ScopedInfraValue],
    max_usd_micros: int,
    max_wall_ms: int,
    max_steps: int,
    metadata: dict[str, Any],
) -> EpisodeSpec:
    """Pin everything the manifest identifies the run by.

    The dependency plan carries one compute requirement because the plan must
    be non-empty and the preflight must have a receipt per requirement; the
    adapter's own ``inspect_requirements`` is what actually gates secrets and
    infra, and the runner fails closed on those before allocation.
    """

    plan = DependencyPlan(
        release_id=benchmark_release,
        task_id=task_id,
        attempt_id=episode_id,
        requirements=(
            ComputeRequirement(
                requirement_id="compute",
                necessity="required",
                reportability="required",
                cpu_class="standard",
                memory_class="small",
                disk_bytes=0,
            ),
        ),
    )
    receipts = (record_preflight(plan, "compute", status="pass", duration_ms=0),)
    preflight = seal_preflight(plan, receipts, RedactionSet.from_resolved_values(()))
    budget = _budget(
        episode_id, max_usd_micros=max_usd_micros, max_wall_ms=max_wall_ms, max_steps=max_steps
    )
    reservation = reserve_budget(
        budget,
        "episode",
        [ResourceAmount(resource=resource, value=1) for resource in BUDGET_RESOURCES],
    )
    config_source = json.dumps(
        {
            "max_steps": max_steps,
            "max_usd_micros": max_usd_micros,
            "max_wall_ms": max_wall_ms,
            "route": route.model_dump(mode="json"),
            "selector": selector.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    return EpisodeSpec(
        episode_id=episode_id,
        task_id=task_id,
        task_digest=_task_digest(selector, task_id),
        input_digest=selector.workspace_digest,
        benchmark_id=benchmark_id,
        benchmark_release=benchmark_release,
        environment_digest=selector.release_digest,
        environment_release=selector.version,
        config_digest=_digest(config_source),
        harness_version=_harness_version(),
        harness_git_revision=_harness_git_revision(),
        requested_route=route,
        dependency_plan=plan,
        budget=budget,
        preflight=preflight,
        reservations=(reservation,),
        fallback_policy="forbid",
        secret_refs=tuple(SecretRef(name=name) for name in secret_refs),
        infra_values=tuple(infra_values),
        metadata=metadata,
    )


def build_config(run_root: Path, *, max_steps: int, max_cycle_usd_micros: int | None) -> Any:
    """Roots under ONE durable directory; timeouts sized for a real worker."""

    refuse_volatile_root(run_root, label="run root")
    evidence = run_root / "evidence"
    artifacts = run_root / "artifacts"
    rescue = run_root / "rescue"
    for path in (evidence, artifacts, rescue):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return EpisodeConfig(
        evidence_root=evidence,
        artifact_root=artifacts,
        rescue_root=rescue,
        max_steps=max_steps,
        # A real AWS allocation waits on instance readiness (up to 10 min in
        # the provider); the reset timeout must cover that or the runner
        # abandons an episode whose instance is still coming up.
        prepare_timeout=120.0,
        reset_timeout=900.0,
        step_timeout=180.0,
        score_timeout=300.0,
        cleanup_timeout=120.0,
        handshake_timeout=60.0,
        max_cycle_cost_micros=max_cycle_usd_micros,
    )


def _open_store(config_dir: Path | None) -> tuple[Any, Any]:
    """(auth_store, credential_manager) exactly as the CLI builds them.

    ``cli._build_auth_stack`` is the reference: ``CredentialManager`` over the
    config dir feeds ``AuthStore``. Imported lazily so the script's import
    graph stays free of the application until a store is actually opened.
    """

    from local_operator.credentials import CredentialManager
    from local_operator.paths import config_dir as default_config_dir
    from local_operator.providers.auth_store import AuthStore

    manager = CredentialManager(config_dir or default_config_dir())
    return AuthStore(credential_manager=manager), manager


def _model_client(
    *,
    auth_store: Any,
    settings: dict[str, Any],
    provider: str,
    model: str,
    route: RouteIdentity,
    artifact_root: Path,
    episode_id: str,
    keep_recent_frames: int,
) -> Any:
    from local_operator.evaluation.runner.provider_client import (
        create_provider_model_client,
    )
    from local_operator.model.configure import build_model_spec

    return create_provider_model_client(
        auth_store=auth_store,
        settings=settings,
        route=route,
        model_spec=build_model_spec(provider, model),
        artifact_root=artifact_root,
        episode_id=episode_id,
        fallback_policy="forbid",
        keep_recent_frames=keep_recent_frames,
    )


def _outcome_json(outcome: EpisodeOutcome) -> dict[str, Any]:
    data = asdict(outcome)
    data["bundle_root"] = str(outcome.bundle_root) if outcome.bundle_root else None
    data["score"] = outcome.score.model_dump(mode="json") if outcome.score else None
    return data


def _parse_infra(values: Sequence[str], purpose: str) -> tuple[ScopedInfraValue, ...]:
    out: list[ScopedInfraValue] = []
    for item in values:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise SystemExit(f"--infra expects NAME=VALUE, got {item!r}")
        # ``purpose`` is validated by the model against the closed InfraPurpose
        # vocabulary; argparse hands it over as a plain string.
        out.append(
            ScopedInfraValue.model_validate({"name": name, "purpose": purpose, "value": value})
        )
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--selector", required=True, type=Path, help="AdapterSelector JSON file")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--route", required=True, type=_parse_route, help="<provider>/<model>")
    parser.add_argument("--run-root", required=True, type=Path, help="durable; never /tmp")
    parser.add_argument("--benchmark-id", default="osworld-v2")
    parser.add_argument("--benchmark-release", default="osworld-v2-2026.08.08")
    parser.add_argument("--episode-id", default=None, help="default: ep-<12 hex>")
    parser.add_argument(
        "--secret",
        action="append",
        default=None,
        metavar="NAME",
        help=f"secret ref the task needs (repeatable; default {list(_DEFAULT_SECRET_REFS)})",
    )
    parser.add_argument(
        "--secret-env",
        action="append",
        default=[],
        metavar="NAME",
        help="read this secret from the process environment instead of the store",
    )
    parser.add_argument("--infra", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--infra-purpose", default="benchmark_compute")
    parser.add_argument("--config-dir", type=Path, default=None, help="lop config dir")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-usd", type=float, default=0.50, help="provider spend cap")
    parser.add_argument("--max-wall-s", type=int, default=1800)
    parser.add_argument("--max-cycle-usd", type=float, default=None, help="per-cycle cap")
    parser.add_argument("--keep-recent-frames", type=int, default=3)
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="never open the credential store: every secret must be on --secret-env "
        "and the model client is built without one (test/fake routes only)",
    )
    parser.add_argument(
        "--model-client",
        default="provider",
        choices=("provider", "scripted-finish"),
        help="'scripted-finish' issues one finish action with no provider "
        "(for proving the script against a fake adapter; never for a result)",
    )
    return parser


class _ScriptedFinish:
    """A model that finishes on its first turn; for exercising the script only.

    It reports the REQUESTED route as served so the bundle seals comparable;
    it never calls a provider, so a run with it is a proof of the plumbing
    (spawn, secrets, frames, seal) and never a benchmark result.
    """

    def __init__(self, route: RouteIdentity) -> None:
        self._route = route

    async def decide(self, observation: Any, history: Sequence[Any]) -> Any:
        from local_operator.evaluation.protocol import ActionBatch
        from local_operator.evaluation.runner.model import ModelDecision

        del history
        batch = ActionBatch.model_validate(
            {
                "protocol_version": "1.0",
                "task_id": observation.task_id,
                "episode_id": observation.episode_id,
                "observation_id": observation.observation_id,
                "actions": [
                    {
                        "kind": "finish",
                        "observation_id": observation.observation_id,
                        "status": "done",
                        "reason": "scripted finish from scripts/run_episode.py",
                    }
                ],
            },
            strict=True,
        )
        return ModelDecision(action_batch=batch, route=self._route)


async def run(args: argparse.Namespace) -> int:
    selector = AdapterSelector.model_validate(json.loads(args.selector.read_text()))
    provider, model = args.route
    route = _route_identity(provider, model)
    episode_id = args.episode_id or f"ep-{uuid.uuid4().hex[:12]}"
    secret_refs = tuple(args.secret) if args.secret is not None else _DEFAULT_SECRET_REFS
    max_usd_micros = int(round(args.max_usd * 1_000_000))
    max_cycle = int(round(args.max_cycle_usd * 1_000_000)) if args.max_cycle_usd else None

    try:
        config = build_config(
            args.run_root, max_steps=args.max_steps, max_cycle_usd_micros=max_cycle
        )
    except VolatileRootError as error:
        print(str(error), file=sys.stderr)
        return EXIT_PREFLIGHT

    auth_store: Any = None
    manager: Any = None
    store_resolver: SecretResolver | None = None
    if not args.no_store:
        from local_operator.evaluation.runner.host_secrets import (
            CredentialStoreResolver,
        )

        auth_store, manager = _open_store(args.config_dir)
        store_resolver = CredentialStoreResolver(manager)
    resolver = _LayeredResolver(args.secret_env, store_resolver)

    spec = build_spec(
        episode_id=episode_id,
        selector=selector,
        task_id=args.task_id,
        route=route,
        benchmark_id=args.benchmark_id,
        benchmark_release=args.benchmark_release,
        secret_refs=secret_refs,
        infra_values=_parse_infra(args.infra, args.infra_purpose),
        max_usd_micros=max_usd_micros,
        max_wall_ms=args.max_wall_s * 1000,
        max_steps=args.max_steps,
        metadata={
            "harness_version": _harness_version(),
            # The unfolded route, verbatim, beside the folded identity the
            # manifest seals: ``route_model_id`` is the exact id the provider
            # was asked for.
            "route": f"{provider}/{model}",
            "route_provider_id": provider,
            "route_model_id": model,
            "script": "scripts/run_episode.py",
        },
    )

    if args.model_client == "scripted-finish":
        model_client: Any = _ScriptedFinish(route)
    else:
        if auth_store is None:
            print(
                "--model-client provider needs the credential store (drop --no-store)",
                file=sys.stderr,
            )
            return EXIT_PREFLIGHT
        model_client = _model_client(
            auth_store=auth_store,
            settings={},
            provider=provider,
            model=model,
            route=route,
            artifact_root=config.artifact_root,
            episode_id=episode_id,
            keep_recent_frames=args.keep_recent_frames,
        )

    runner = EpisodeRunner(
        spec,
        config,
        selector=selector,
        model=model_client,
        secrets=resolver,
        launch=AdapterSupervisor.launch,
    )
    try:
        outcome = await runner.run()
    finally:
        if auth_store is not None:
            auth_store.close()

    print(json.dumps(_outcome_json(outcome), indent=2, sort_keys=True))
    if outcome.status == "failed_pre_bundle" and (outcome.diagnostic or "").startswith(
        "MissingSecret"
    ):
        # The runner's diagnostic already names only the ref.
        print(outcome.diagnostic, file=sys.stderr)
        return EXIT_PREFLIGHT
    return EXIT_OK if outcome.status == "completed" else EXIT_EPISODE


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

"""Tier 1: one ECS Fargate task per run, lifted out of the POC script.

WHY IT IS NOT THE SCRIPT. ``scripts/remote_agents_poc.py`` stays in ``scripts/``
as the POC's own evidence tooling — its ``report``/``verify``/``selftest`` half is
not product code and never will be — and what is lifted here is only its
run/status/stop logic. What the provider adds is what a driver script never needs:
a Protocol, a status record, a cancel, and an account guard that belongs to the
provider rather than to whoever happens to call it (§4.1). The script's behaviour
is deliberately unchanged: its recorded evidence depends on it.

THE TWO RULES THE POC LEARNED THE HARD WAY, both kept verbatim:

* the account assertion runs FIRST, before a single mutating call, because every
  hard rule about this sandbox starts with "the account is 325492156725";
* the overrides document is refused at 8 KiB rather than discovered at RunTask
  time, where ECS answers a 400 that names no field (measured).

``status()`` IS ONE ``describe_tasks``. The POC's poll loop — a 1 s grid with a
2 h + 10 min client deadline — belongs to the caller that wants to wait, which is
``drain()`` here and the CLI's run loop in the next slice: a ``status()`` that
blocks for two hours is not a status call.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, cast

from local_operator.cloud.aws import (
    EXPECTED_POC_ACCOUNT,
    REGION,
    AwsCredentials,
    CloudBlobs,
    FargateClients,
    build_clients,
    sdk_exceptions,
    verify_account,
)
from local_operator.cloud.provider import (
    CloudRefused,
    ComputeStatus,
    DrainOutcome,
    ProviderCapacity,
    ProviderPhase,
)
from local_operator.cloud.types import PoolRequest
from local_operator.cloud.workload import task_environment

#: Every task this provider creates carries these, so a bad run is still found by
#: the operator's tag inventory instead of becoming an untagged orphan. The
#: inventory itself is NOT the provider's (§4.1): `resourcegroupstaggingapi` has no
#: resource-level scoping, so it stays an operator-identity tool and `lop cloud
#: list` reads the ledger.
POC_TAGS: dict[str, str] = {"lop-poc": "true", "owner": "lopdev"}

#: The container's name in the task definition, and the only one overrides may name.
CONTAINER_NAME = "agent"

#: The two artifacts the image writes, by the names it writes them under.
PROBES_OBJECT = "probes.json"
RESULTS_OBJECT = "results.tar.gz"

#: ECS rejects an overrides document above 8 KiB. Asserted rather than discovered
#: at RunTask time, because the failure mode there is a 400 with no field named.
OVERRIDES_LIMIT_BYTES = 8192

#: The client deadline, 2 h + 10 min (§9.2 step 3): "StopTask if exceeded".
RUN_DEADLINE_SECONDS = 2 * 60 * 60 + 10 * 60

#: The POC's poll grid.
DESCRIBE_POLL_SECONDS = 1.0

#: ``describe_tasks``'s lastStatus, as the seam's own vocabulary.
_PHASES: dict[str, ProviderPhase] = {
    "PROVISIONING": "pending",
    "PENDING": "pending",
    "ACTIVATING": "pending",
    "RUNNING": "running",
    "DEACTIVATING": "stopped",
    "STOPPING": "stopped",
    "STOPPED": "stopped",
}


@dataclass(frozen=True)
class FargateTargets:
    """What the stack outputs say this provider has to talk to.

    Built by the CLI from ``pulumi stack output --json`` (§4.2): the provider never
    shells out to pulumi, and it never reads an ``argparse`` namespace either — it
    is handed values.
    """

    bucket: str
    subnets: tuple[str, ...]
    security_group: str
    secret_arn: str = ""
    queue_arn: str = ""
    log_group: str = ""


@dataclass(frozen=True)
class _Submitted:
    """What this provider remembers about a task it created.

    Remembered rather than re-derived because ``drain(provider_ref)`` is handed
    nothing else (§3.2's signature), and the checkpoint marker it waits for is the
    results object at a key only this provider knows. A provider that has lost this
    (a restart) refuses to drain rather than pretending it watched: see
    :meth:`FargateProvider.drain`.
    """

    request_id: str
    probes_url: str
    results_url: str
    results_key: str


class FargateProvider:
    """The ECS Fargate backend: ``name == "fargate"``."""

    name = "fargate"

    def __init__(
        self,
        credentials: AwsCredentials,
        *,
        region: str = REGION,
        cluster: str,
        task_definition: str,
        targets: FargateTargets,
        expected_account: str = EXPECTED_POC_ACCOUNT,
        run_deadline_s: float = RUN_DEADLINE_SECONDS,
        poll_interval_s: float = DESCRIBE_POLL_SECONDS,
        clients: FargateClients | None = None,
        blobs: CloudBlobs | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._credentials = credentials
        self._region = region
        self._cluster = cluster
        self._task_definition = task_definition
        self._targets = targets
        self._expected_account = expected_account
        self._run_deadline_s = run_deadline_s
        self._poll_interval_s = poll_interval_s
        #: Injected on EVERY call, so the provider never reaches the SDK's own
        #: session discovery and a test never needs credentials.
        self._clients = clients if clients is not None else build_clients(credentials, region)
        self._blobs = (
            blobs if blobs is not None else CloudBlobs(s3=self._clients.s3, bucket=targets.bucket)
        )
        #: The blocking wait, injectable for the same reason ``AwsProvider._sleep``
        #: is: a drain's deadline must be exercised without waiting for it.
        self._sleep = sleep
        #: And the CLOCK is injectable for the same reason, which is why it is a
        #: parameter rather than a patched ``time.monotonic``: ``drain`` runs on the
        #: event loop, and monkeypatching the loop's own clock to test a timeout is
        #: how a test hangs instead of failing.
        self._clock = clock
        self._submitted: dict[str, _Submitted] = {}
        self._account_arn: str | None = None

    # -- the Protocol ------------------------------------------------------

    async def describe(self) -> ProviderCapacity:
        """Resolve the task definition, and say so rather than raising.

        A stale ARN is a refusal BY NAME, not an exception: ``lop cloud status``
        must be able to SAY that the stack is gone, and the POC's reason for
        checking at all is that a ``pulumi up`` which changes a task definition
        REPLACES it, retiring the previous revision into the unhelpful "the task
        definition is inactive" from RunTask.
        """
        detail = {
            "cluster": self._cluster,
            "task_definition": self._task_definition,
            "region": self._region,
        }
        ecs = self._definition_client()
        errors = sdk_exceptions()
        try:
            definition = await asyncio.to_thread(
                ecs.describe_task_definition, taskDefinition=self._task_definition
            )
        except errors.ClientError as error:
            # A retired revision and a role that may not read definitions arrive
            # here as the same shape, so the code is reported rather than guessed
            # away: the sentence says which call failed.
            return ProviderCapacity(
                provider=self.name,
                tier="fargate",
                ready=False,
                detail={**detail, "reason": "task_definition_inactive", "error": _code_of(error)},
            )
        task_definition = definition.get("taskDefinition") or {}
        status = str(task_definition.get("status") or "").upper()
        if status != "ACTIVE":
            return ProviderCapacity(
                provider=self.name,
                tier="fargate",
                ready=False,
                detail={**detail, "reason": "task_definition_inactive", "status": status},
            )
        return ProviderCapacity(
            provider=self.name,
            tier="fargate",
            ready=True,
            detail={
                **detail,
                "revision": str(task_definition.get("revision") or ""),
                "family": str(task_definition.get("family") or ""),
                "image": _container_image(task_definition),
            },
        )

    async def submit(self, request: PoolRequest) -> str:
        """Create the task, and return its ARN. The only method that creates anything.

        The order is the point: refusals that need no API call (budget, tier,
        network), then the ACCOUNT assertion, then the ACTIVE definition, then the
        presigned URLs, then the size check, and only then RunTask. Every step
        before the last one can refuse without leaving anything behind.
        """
        self._refuse_unusable(request)
        await self._ensure_account()
        capacity = await self.describe()
        if not capacity.ready:
            raise CloudRefused(
                "provider_unavailable",
                f"refusing to submit: {capacity.detail.get('reason', 'the stack is not ready')} "
                f"for {self._cluster} in {self._region}",
            )
        run_id = request.request_id
        probes_key = self._blobs.key(run_id, PROBES_OBJECT)
        results_key = self._blobs.key(run_id, RESULTS_OBJECT)
        probes_url = await asyncio.to_thread(self._blobs.presign_put, probes_key)
        results_url = await asyncio.to_thread(self._blobs.presign_put, results_key)
        environment = task_environment(
            request,
            probes_url=probes_url,
            results_url=results_url,
            model_secret_arn=self._targets.secret_arn,
        )
        overrides = {"containerOverrides": [{"name": CONTAINER_NAME, "environment": environment}]}
        overrides_bytes = len(json.dumps(overrides).encode())
        if overrides_bytes >= OVERRIDES_LIMIT_BYTES:
            raise CloudRefused(
                "overrides_too_large",
                f"the overrides document is {overrides_bytes} bytes, at or over the "
                f"{OVERRIDES_LIMIT_BYTES}-byte ECS limit; shorten the prompt",
            )
        response = await asyncio.to_thread(
            self._clients.ecs.run_task,
            cluster=self._cluster,
            taskDefinition=self._task_definition,
            launchType="FARGATE",
            platformVersion="LATEST",
            count=1,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": list(self._targets.subnets),
                    "securityGroups": [self._targets.security_group],
                    "assignPublicIp": "ENABLED",
                }
            },
            overrides=overrides,
            tags=[
                {"key": key, "value": value}
                for key, value in {**POC_TAGS, "run-id": run_id}.items()
            ],
            # enableECSManagedTags + propagateTags TASK_DEFINITION: a run that fails
            # to start still carries lop-poc, so the inventory finds it instead of
            # it becoming an untagged orphan.
            enableECSManagedTags=True,
            propagateTags="TASK_DEFINITION",
        )
        failures = response.get("failures") or []
        if failures:
            raise CloudRefused("provider_unavailable", f"RunTask failed: {failures}")
        tasks = response.get("tasks") or []
        if not tasks:
            raise CloudRefused("provider_unavailable", "RunTask returned no task")
        provider_ref = str(tasks[0]["taskArn"])
        self._submitted[provider_ref] = _Submitted(
            request_id=run_id,
            probes_url=probes_url,
            results_url=results_url,
            results_key=results_key,
        )
        return provider_ref

    async def status(self, provider_ref: str) -> ComputeStatus:
        """One ``describe_tasks``. Never mutates, and never waits.

        An EMPTY task list is ``pending`` and not ``absent``: ECS cannot describe a
        task for the first fractions of a second after RunTask, and the POC's own
        sleep-and-continue exists because of it. The honest reading of "I cannot see
        it yet" is not "it is gone", so ``absent`` is reserved for the case ECS
        states out loud — a failure entry naming THIS arn as missing, which is what
        it answers once a stopped task ages out of its inventory.
        """
        response = await asyncio.to_thread(
            self._clients.ecs.describe_tasks,
            cluster=self._cluster,
            tasks=[provider_ref],
            # Kept from the POC's call shape so the two can be diffed as behaviour
            # rather than as arguments: the tags are what tell a human which run a
            # task belongs to while it is still live.
            include=["TAGS"],
        )
        tasks = response.get("tasks") or []
        if not tasks:
            if _names_missing(response.get("failures") or [], provider_ref):
                return ComputeStatus(
                    provider=self.name,
                    provider_ref=provider_ref,
                    phase="absent",
                    detail={"reason": "missing"},
                )
            return ComputeStatus(
                provider=self.name,
                provider_ref=provider_ref,
                phase="pending",
                detail={"reason": "not_yet_describable"},
            )
        task = tasks[0]
        containers = task.get("containers") or []
        container = containers[0] if containers else {}
        exit_code = container.get("exitCode")
        last_status = str(task.get("lastStatus") or "")
        return ComputeStatus(
            provider=self.name,
            provider_ref=provider_ref,
            phase=_PHASES.get(last_status, "pending"),
            exit_code=None if exit_code is None else int(exit_code),
            stop_reason=str(task.get("stoppedReason") or container.get("reason") or ""),
            started_at=_epoch(task.get("startedAt")),
            stopped_at=_epoch(task.get("stoppedAt")),
            detail={
                "last_status": last_status,
                "stop_code": str(task.get("stopCode") or ""),
                "container_reason": str(container.get("reason") or ""),
            },
        )

    async def cancel(self, provider_ref: str, *, reason: str) -> None:
        """``stop_task``. Idempotent at ECS's end: a stopped task stays stopped."""
        await asyncio.to_thread(
            self._clients.ecs.stop_task,
            cluster=self._cluster,
            task=provider_ref,
            reason=reason,
        )

    async def drain(self, provider_ref: str, *, deadline_s: float) -> DrainOutcome:
        """Wait for the container's checkpoint marker, then stop the task.

        What the wait is FOR. ECS sends SIGTERM to PID 1 and SIGKILLs after the task
        definition's ``stopTimeout``, so asking a task to stop is not the same as it
        having flushed anything. The marker this waits for is the container's own:
        the results object appearing at the key :meth:`submit` presigned. When the
        deadline passes the task is STOPPED anyway, with a reason that NAMES the
        deadline — the POC's ``poll_to_stopped`` overdue path, which is "a stop, not
        a surrender" — and ``drained`` stays ``False``, because a provider that was
        asked for a checkpoint it never confirmed must not read as success.
        """
        status = await self.status(provider_ref)
        if status.stopped:
            return DrainOutcome(
                drained=False, waited_s=0.0, reason="already_stopped", checkpoint={}
            )
        submitted = self._submitted.get(provider_ref)
        if submitted is None:
            # No marker to watch, so no honest way to wait for one: this provider
            # did not create the task in this process. Stop it and say so rather
            # than spending the container's budget on a checkpoint nothing can see.
            await self.cancel(provider_ref, reason="lop drain: no checkpoint marker recorded")
            return DrainOutcome(
                drained=False, waited_s=0.0, reason="provider_refused", checkpoint={}
            )
        # The run's own client deadline is a ceiling on any drain inside it: a drain
        # that outlasted it would fight the StopTask that is already scheduled.
        deadline_s = min(deadline_s, self._run_deadline_s)
        started = self._clock()
        deadline = started + deadline_s
        checkpointed = False
        while self._clock() < deadline:
            if await asyncio.to_thread(self._blobs.object_present, submitted.results_key):
                checkpointed = True
                break
            await asyncio.to_thread(self._sleep, self._poll_interval_s)
        waited_s = self._clock() - started
        checkpoint = {
            "branch": f"lop/{submitted.request_id}",
            "commit": "",
            "session_id": "",
            "artifacts": submitted.results_url,
        }
        if checkpointed:
            return DrainOutcome(
                drained=True, waited_s=waited_s, reason="checkpointed", checkpoint=checkpoint
            )
        await self.cancel(provider_ref, reason=f"lop drain deadline ({deadline_s:g}s)")
        return DrainOutcome(drained=False, waited_s=waited_s, reason="deadline", checkpoint={})

    # -- internals ---------------------------------------------------------

    def _refuse_unusable(self, request: PoolRequest) -> None:
        """The refusals that need no API call, before anything is created.

        ``budget_usd`` of 0 is refused rather than defaulted (§3.3): the cap is the
        user's statement about what this run may cost, and inventing one is exactly
        the decision a provider must not make for them.
        """
        if request.tier != "fargate":
            raise CloudRefused(
                "tier_unknown",
                f"this provider runs tier 'fargate', not {request.tier!r}",
            )
        if request.budget_usd <= 0.0:
            raise CloudRefused(
                "budget_missing",
                "refusing to submit: budget_usd is 0, and a run needs a cap to be "
                "authorised against",
            )
        if request.network_id:
            raise CloudRefused(
                "network_unbound",
                "cloud membership is not built; run without a network (§9.3)",
            )

    def _definition_client(self) -> Any:
        """The client allowed to read task definitions (see ``FargateClients``)."""
        return (
            self._clients.definitions
            if self._clients.definitions is not None
            else self._clients.ecs
        )

    async def _ensure_account(self) -> str:
        """Assert the account once per provider, before any mutating call (§4.1)."""
        if self._account_arn is None:
            self._account_arn = await asyncio.to_thread(
                verify_account, self._clients, self._expected_account
            )
        return self._account_arn


def _code_of(error: BaseException) -> str:
    """The AWS error code out of a ``ClientError``'s response, or its class name."""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        if code:
            return str(code)
    return type(error).__name__


def _container_image(task_definition: dict[str, Any]) -> str:
    containers = task_definition.get("containerDefinitions") or [{}]
    return str(containers[0].get("image") or "")


def _names_missing(failures: list[Any], provider_ref: str) -> bool:
    """Whether ECS said out loud that THIS reference is missing."""
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        if str(failure.get("arn") or "") != provider_ref:
            continue
        if str(failure.get("reason") or "").upper() == "MISSING":
            return True
    return False


def _epoch(value: Any) -> float | None:
    """A datetime ECS returned, as a POSIX timestamp (or ``None``).

    ECS's timestamps arrive as ``datetime``s and the record keeps floats, so the
    conversion happens here, once, rather than at every reader.
    """
    timestamp = getattr(value, "timestamp", None)
    if not callable(timestamp):
        return None
    return float(cast(Callable[[], Any], timestamp)())

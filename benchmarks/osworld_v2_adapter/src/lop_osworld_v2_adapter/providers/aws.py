"""AwsProvider: the one module in this adapter that spends money.

It launches the OSWorld guest on EC2 WITH boto3 DIRECTLY, then hands the
instance id to upstream's ``DesktopEnv(path_to_vm=<id>)`` so every guest
interaction (setup, screenshots, actions, evaluation) is upstream code called
unmodified. It does NOT use OSWorld's own ``AWSVMManager._allocate_vm``, and
the reasons are each a released-contract requirement, not a preference:

- OSWorld's ``run_instances`` carries no ``ClientToken`` (manager.py:222-266),
  so a retried ``reset_start`` double-allocates. Ours is minted from the
  episode id (``ProvisioningPlan.client_token``).
- OSWorld tags only ``Name`` and only when an env var is set (manager.py:254).
  The cleanup contract (``cleanup.py``) resolves the instance BY TAG from a
  persisted descriptor, so ``lop:episode`` and ``lop:adapter`` must be in
  ``TagSpecifications`` at creation — and the operator's TTL role is scoped by
  ``aws:ResourceTag/lop:adapter``, so without the tag the lease cannot fire.
- OSWorld treats EventBridge TTL failure as a logged WARNING
  (manager.py:274-276). The previous pilot lost paid instances to SIGKILL, a
  sleeping laptop and a ``/tmp`` purge because teardown was foreground-only.
  Here the TTL lease is created immediately after ``run_instances`` and its
  failure is FATAL to ``allocate`` (terminate what was launched, then raise).
- OSWorld installs process-wide SIGINT/SIGTERM handlers inside ``_allocate_vm``
  (manager.py:181-206), which is unacceptable inside an asyncio worker whose
  parent owns signal delivery.
- ``DesktopEnv.close()`` calls ``terminate_instances`` and returns without
  confirming (provider.py:279-285). It is NEVER called; ``terminate`` here
  polls ``describe_instances`` until the state is ``terminated`` and reports
  ``terminate-unconfirmed`` (rescue-required) if it is not reached in time.

Credentials arrive as ``AwsCredentials`` (from ``ResetStartParams.secrets`` /
``BeginRescueParams.secrets``), never from ``os.environ``: the worker's
environment is stripped and MUST stay that way. Every client is built with an
explicit ``region_name`` — the operator's default profile region is not
us-east-1, and a client that fell back to the profile would launch the
release-pinned AMI in a region where it does not exist.

Every boto3 / HTTP call is blocking and runs under ``asyncio.to_thread`` so the
worker's event loop keeps answering the parent's RPC while a waiter sleeps.

``allocate`` also runs one GUEST PREPARATION step between readiness and
upstream's reset: the released AMI ships ~93% full and its own snapd fills the
remainder on a clock, which destroyed 7 of 8 episodes in a 424-466s window. See
``guest_disk`` for the measurements, for why ffmpeg was NOT the cause, and for
why that step can never fail an allocation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from lop_osworld_v2_adapter import guest_disk, scoring
from lop_osworld_v2_adapter.cleanup import (
    EVIDENCE_INSTANCE_ABSENT,
    EVIDENCE_INSTANCE_TERMINATED,
    EVIDENCE_SCHEDULE_ABSENT,
    EVIDENCE_SCHEDULE_DELETE_FAILED,
    EVIDENCE_SCHEDULE_DELETED,
    EVIDENCE_TERMINATE_DENIED,
    EVIDENCE_TERMINATE_UNCONFIRMED,
)
from lop_osworld_v2_adapter.provisioning import ProvisioningPlan
from lop_osworld_v2_adapter.taskfile import TaskDescriptor

# The tag every resource this adapter creates carries. It is both the leak
# detector's filter (``audit``) and the condition key on the operator's TTL
# role, so it must match ``provisioning.resolve`` exactly.
ADAPTER_TAG_KEY = "lop:adapter"
ADAPTER_TAG_VALUE = "osworld-v2"
EPISODE_TAG_KEY = "lop:episode"

# States in which an instance still exists and can be terminated. ``terminated``
# is deliberately absent: an instance already there is ``instance-absent`` for
# cleanup purposes, and AWS keeps terminated instances visible for an hour.
LIVE_STATES = ("pending", "running", "stopping", "stopped", "shutting-down")

# Default lease when the budget carries no wall cap (DESIGN §B.5): long enough
# for a 500-step episode at ~3 s/action plus setup, short enough that a dead
# controller does not bill for OSWorld's own 5-hour default.
DEFAULT_TTL_SECONDS = 7200
# Slack added to the wall budget so the lease never fires on a legitimately
# slow but in-budget episode.
TTL_SLACK_SECONDS = 15 * 60

# The EventBridge Scheduler universal target for terminating instances.
_TERMINATE_TARGET_ARN = "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances"

# The OSWorld guest control port and the readiness endpoint upstream's own
# ``SetupController.ensure_ready`` probes (setup.py:58-70).
GUEST_PORT = 5000
_READY_PATH = "/terminal"
# The guest's command endpoint. ``shell: false`` means the server execs the argv
# directly, so a pipeline arrives as an explicit ``bash -c`` -- the same shape
# upstream's own SetupController posts (setup.py:418).
_EXECUTE_PATH = "/execute"

# Where the episode's guest-preparation evidence is written, inside the
# episode-owned cache root the adapter mints in ``reset_start``. It sits there
# rather than in the artifact root because the bundle verifier refuses any entry
# in the artifact root that is not a digest-named artifact.
GUEST_PREPARATION_FILENAME = "guest-preparation.json"


class AllocationError(RuntimeError):
    """``allocate`` could not reach a fully-leased, ready guest."""


class ReadinessTimeout(AllocationError):
    """The instance ran but the guest service never answered in time."""


@dataclass(frozen=True)
class AwsCredentials:
    """The resolved AWS secrets. Never logged, never placed in the environment."""

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - defensive against accidental echo
        return "AwsCredentials(<redacted>)"


@dataclass
class _Clients:
    """The injectable I/O surface, so tests use botocore's Stubber.

    ``ec2`` and ``scheduler`` are boto3 clients; ``http_get`` takes a URL and
    a timeout and returns the HTTP status code (or raises); ``http_post_json``
    posts a JSON body and returns the decoded response (or raises). Production
    builds all four from the credentials; tests hand in stubbed clients and
    fake probers.

    ``http_post_json`` is REQUIRED rather than optional-with-a-default: it is
    what drives the guest's control server for pre-episode disk reclamation
    (``guest_disk``), and a default that quietly did nothing would make that
    preparation silently absent in exactly the builds nobody checked.
    """

    ec2: Any
    scheduler: Any
    http_get: Callable[[str, float], int]
    http_post_json: Callable[[str, dict[str, Any], float], dict[str, Any]]


def build_clients(credentials: AwsCredentials, region: str) -> _Clients:
    """Construct real clients from the resolved secrets.

    ``region_name`` is passed explicitly on every client. The session is
    built from the credential VALUES, so boto3's own provider chain (env vars,
    ``~/.aws``, instance metadata) is never consulted — which is what keeps
    the worker's stripped environment authoritative.
    """

    import requests  # type: ignore[import-not-found]
    from boto3.session import Session  # type: ignore[import-not-found]

    session = Session(
        aws_access_key_id=credentials.access_key_id,
        aws_secret_access_key=credentials.secret_access_key,
        aws_session_token=credentials.session_token,
        region_name=region,
    )

    def http_get(url: str, timeout: float) -> int:
        return int(requests.get(url, timeout=timeout).status_code)

    def http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, dict):
            raise ValueError("guest control server returned a non-object body")
        return decoded

    return _Clients(
        ec2=session.client("ec2", region_name=region),
        scheduler=session.client("scheduler", region_name=region),
        http_get=http_get,
        http_post_json=http_post_json,
    )


def ttl_seconds_for(wall_budget_ms: int | None, *, override: int | None = None) -> int:
    """Derive the lease length from the episode's wall budget.

    An explicit operator override (``OSWORLD_TTL_SECONDS`` infra) wins; a
    capped wall budget produces ``budget + slack``; an uncapped one falls back
    to ``DEFAULT_TTL_SECONDS``. The result is never below the slack itself, so
    a tiny budget cannot produce a lease that fires during boot.
    """

    if override is not None:
        return max(int(override), TTL_SLACK_SECONDS)
    if wall_budget_ms is None:
        return DEFAULT_TTL_SECONDS
    return max(int(wall_budget_ms // 1000) + TTL_SLACK_SECONDS, TTL_SLACK_SECONDS)


def _episode_id_from_ref(instance_ref: str) -> str:
    """``lop-ep-<episode_id>`` -> ``<episode_id>``; the ref shape from cleanup.py."""

    prefix = "lop-ep-"
    if not instance_ref.startswith(prefix):
        raise ValueError(f"instance ref {instance_ref!r} is not a lop-ep- tag ref")
    return instance_ref[len(prefix) :]


def _tag_filters(episode_id: str | None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [
        {"Name": f"tag:{ADAPTER_TAG_KEY}", "Values": [ADAPTER_TAG_VALUE]},
        {"Name": "instance-state-name", "Values": list(LIVE_STATES)},
    ]
    if episode_id is not None:
        filters.insert(0, {"Name": f"tag:{EPISODE_TAG_KEY}", "Values": [episode_id]})
    return filters


def _instances(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]


def _error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return ""


class UpstreamAllocationRefused(RuntimeError):
    """Upstream OSWorld tried to allocate, revert, or release an instance.

    Every upstream launch path (``AWSVMManager._allocate_vm``,
    ``AWSProvider.revert_to_snapshot``) issues ``run_instances`` with NO
    ``ClientToken``, NO ``lop:adapter``/``lop:episode`` tag (``Name`` only,
    and only when ``AWS_INSTANCE_NAME`` is set) and NO TTL schedule (our
    ``ENABLE_TTL=false``). An instance it created would be invisible to the
    tag audit, unreachable by descriptor-driven rescue, and unleased -- the
    exact leak this provider exists to prevent. ``stop_emulator`` is the
    unconfirmed terminate we also never want. So ``_seal_upstream`` replaces
    every one of those methods on the live env with a raiser, and this is
    what it raises: BEFORE any boto3 call, from inside the worker, surfacing
    to the runner as an adapter error whose cleanup path terminates OUR
    tagged instance by tag and confirms it.
    """


# Files, not logger names, identify a judge failure. ``llm_metrics.py`` logs
# its swallowed exceptions on the SHARED ``desktopenv.env`` logger -- the
# same one ``DesktopEnv.reset``/``evaluate`` use for benign setup-retry
# ERRORs ("Environment setup failed, retrying (1/5)...") -- so filtering by
# logger name either misses the judge (drop ``desktopenv.env``) or trips on a
# good score after a recoverable retry (keep it). ``LogRecord.pathname`` is
# set by the caller's frame and is what actually distinguishes them. The
# pinned checkout is what makes matching a basename safe.
_JUDGE_SOURCE_FILES = frozenset({"llm_metrics.py", "model_client.py"})
# The loggers those files write to; the handler is attached here only.
_JUDGE_LOGGERS = ("desktopenv.eval_model", "llm_metrics", "desktopenv.env")


class _JudgeErrorCapture(logging.Handler):
    """Records ERROR records emitted FROM OSWorld's judge modules.

    ``llm_metrics`` swallows every judge exception into ``0.0`` after a
    ``logger.error("Error in compare_...")`` (llm_metrics.py:176-178). The
    only observable trace of a failed judge call is that log record, so
    capturing it is what lets ``evaluate`` raise ``ScoringUnavailable``
    instead of returning a silent zero. Records from any other file --
    setup retries, "No evaluator configured", metric file-not-found -- are
    deliberately NOT captured: those are either benign or already the
    score's own honest content.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if os.path.basename(record.pathname) not in _JUDGE_SOURCE_FILES:
            return
        try:
            self.messages.append(record.getMessage())
        except Exception:  # pragma: no cover - a broken format must not mask the error
            self.messages.append(record.msg if isinstance(record.msg, str) else "judge error")


class AwsProvider:
    """Satisfies ``EnvironmentProvider`` against a real EC2 guest."""

    def __init__(
        self,
        credentials: AwsCredentials,
        *,
        region: str,
        lease_ref: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        readiness_timeout_s: float = 600.0,
        action_delay_s: float = 3.0,
        terminate_timeout_s: float = 55.0,
        clients: _Clients | None = None,
        desktop_env_factory: Callable[..., Any] | None = None,
        task_factory: Callable[[TaskDescriptor], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._region = region
        self._lease_ref = lease_ref
        self._ttl_seconds = ttl_seconds
        self._readiness_timeout_s = readiness_timeout_s
        self._action_delay_s = action_delay_s
        # Below the release-instance action's 60 s ``timeout_ms`` so the
        # provider reports ``terminate-unconfirmed`` itself rather than having
        # the parent time the whole call out and record nothing.
        self._terminate_timeout_s = terminate_timeout_s
        self._clients = clients if clients is not None else build_clients(credentials, region)
        # Retained ONLY for _install_default_session; never logged (see
        # AwsCredentials.__repr__), dropped with the provider.
        self._credentials = credentials
        self._desktop_env_factory = desktop_env_factory
        self._task_factory = task_factory
        self._sleep = sleep
        self._instance_id: str | None = None
        self._public_ip: str | None = None
        self._env: Any = None
        # The episode-scoped, absolute, workspace-external cache root the
        # adapter mints in reset_start. None until allocate; a rescue
        # (teardown-only) provider never allocates and never needs one.
        self._cache_root: Path | None = None
        self._schedule_created = False

    @classmethod
    def for_teardown(
        cls,
        credentials: AwsCredentials,
        *,
        region: str,
        lease_ref: str,
        clients: _Clients | None = None,
    ) -> "AwsProvider":
        """A provider that can ONLY terminate/delete/describe.

        Built by ``begin_rescue`` from the descriptor's infra values plus the
        freshly delivered secrets. It never allocates, so it needs no plan,
        task, or DesktopEnv — teardown resolves everything by tag.
        """

        return cls(credentials, region=region, lease_ref=lease_ref, clients=clients)

    # ------------------------------------------------------------------
    # allocate
    # ------------------------------------------------------------------

    async def allocate(
        self, plan: ProvisioningPlan, task: TaskDescriptor, *, cache_root: Path
    ) -> None:
        # Typed as required (matching the Protocol), and re-checked at
        # runtime: a DesktopEnv built without a cache root would fall back to
        # upstream's cwd-relative default and re-open the digest-break
        # defect. The type stops a caller that forgets it; this stops one
        # that passes None dynamically.
        if cache_root is None:
            raise AllocationError(
                "allocate needs the adapter's episode cache root (cache_root); "
                "without it upstream writes land in the digest-pinned workspace"
            )
        self._cache_root = cache_root
        if plan.region != self._region:
            raise AllocationError(
                f"plan region {plan.region!r} differs from provider region {self._region!r}"
            )
        instance_id = await asyncio.to_thread(self._run_instance, plan)
        self._instance_id = instance_id
        # THE LEASE IS CREATED BEFORE READINESS, and its failure is fatal. A
        # laptop that dies during the guest's boot is exactly the case the
        # lease exists for; a warning here would be the old pilot's leak.
        try:
            await asyncio.to_thread(self._create_lease, instance_id, plan.scheduler_role_arn)
        except Exception as error:
            await asyncio.to_thread(self._terminate_best_effort, instance_id)
            raise AllocationError(
                f"TTL lease {self._lease_ref!r} could not be created; "
                f"instance {instance_id} was terminated"
            ) from error
        self._schedule_created = True
        self._public_ip = await asyncio.to_thread(self._wait_ready, instance_id)
        # BEFORE upstream's reset, and therefore before the episode's first
        # observation: the guest ships ~93% full and its own snapd fills the
        # rest on a clock, which killed 7 of 8 episodes in a 424-466s window.
        # See ``guest_disk`` for the measurements and for why this cannot fail
        # the allocation.
        await asyncio.to_thread(self._prepare_guest_disk, plan)
        await asyncio.to_thread(self._start_desktop_env, plan, task)

    def _run_guest_command(
        self, command: Sequence[str], timeout: float
    ) -> guest_disk.CommandResult:
        """One command in the guest, through its own control server.

        Raises on any transport failure; ``guest_disk`` catches that and records
        the guest as unreachable. The response shape is upstream's own
        (``{"status", "output", "error", "returncode"}``, setup.py:425-441), and
        a missing ``returncode`` reads as a failure rather than a success: a
        server that did not say the command succeeded did not say it succeeded.
        """

        url = f"http://{self._public_ip}:{GUEST_PORT}{_EXECUTE_PATH}"
        body = self._clients.http_post_json(
            url, {"command": list(command), "shell": False}, timeout
        )
        returncode = body.get("returncode")
        return guest_disk.CommandResult(
            returncode=returncode if isinstance(returncode, int) else 1,
            stdout=str(body.get("output") or ""),
            stderr=str(body.get("error") or ""),
        )

    def _prepare_guest_disk(self, plan: ProvisioningPlan) -> None:
        """Reclaim the guest's root filesystem and record what happened.

        Wrapped in a blanket ``except`` on purpose, on top of ``guest_disk``'s
        own per-step fail-soft: this is a hygiene step, and the one outcome that
        must be impossible is a housekeeping defect destroying an episode that
        would otherwise have run. The report is still written when the
        reclamation itself achieved nothing -- "the guest had N bytes free at the
        start" is the fact a later environment failure is read against.
        """

        report: guest_disk.GuestDiskReport | None = None
        try:
            report = guest_disk.prepare_guest_disk(
                self._run_guest_command,
                client_password=plan.client_password,
                clock=time.monotonic,
            )
        except Exception:
            # ``prepare_guest_disk`` raises nothing by contract; this catches a
            # contract violation (or an injected runner that misbehaves) rather
            # than a normal failure mode, so there is no report to write.
            return
        if self._cache_root is None:
            return
        try:
            (self._cache_root / GUEST_PREPARATION_FILENAME).write_bytes(report.to_json_bytes())
        except OSError:
            # An unwritable cache root is the adapter's problem to surface
            # elsewhere (upstream writes there too); losing the hygiene report
            # must not be the thing that ends the episode.
            pass

    def _run_instance(self, plan: ProvisioningPlan) -> str:
        ec2 = self._clients.ec2
        volume_gb = plan.volume_gb
        if volume_gb is None:
            volume_gb = self._resolve_root_volume_size(plan.ami_id)
        else:
            # A pinned size is checked against the AMI's OWN snapshot before
            # run_instances. AWS refuses a root volume smaller than the
            # snapshot it is restored from, and does it with an
            # InvalidBlockDeviceMapping message that names neither the AMI's
            # size nor the one asked for -- so the operator learns only that
            # something was wrong, mid-launch. Checking here costs one
            # read-only describe_images on the path that already issues one in
            # the None branch, and turns that into a message carrying both
            # numbers. It cannot move to ``provisioning.resolve``: prepare is
            # declarative and issues no I/O at all, which is exactly what lets
            # it run before anything is allocated.
            self._refuse_volume_smaller_than_ami(plan.ami_id, volume_gb)
        tags = [{"Key": key, "Value": value} for key, value in plan.tags]
        response = ec2.run_instances(
            MaxCount=1,
            MinCount=1,
            ImageId=plan.ami_id,
            InstanceType=plan.instance_type,
            EbsOptimized=True,
            InstanceInitiatedShutdownBehavior="terminate",
            ClientToken=plan.client_token,
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "SubnetId": plan.subnet_id,
                    "AssociatePublicIpAddress": True,
                    "Groups": [plan.security_group_id],
                }
            ],
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": volume_gb,
                        "VolumeType": "gp3",
                        "Throughput": 1000,
                        "Iops": 4000,
                        "DeleteOnTermination": True,
                    },
                }
            ],
            # Both the instance AND its root volume carry the tags: the
            # audit lists volumes too, and a volume that outlived its
            # instance (DeleteOnTermination is a default, not a law) would
            # otherwise be invisible to the leak detector.
            TagSpecifications=[
                {"ResourceType": "instance", "Tags": tags},
                {"ResourceType": "volume", "Tags": tags},
            ],
        )
        instances = response.get("Instances", [])
        if len(instances) != 1 or "InstanceId" not in instances[0]:
            raise AllocationError("run_instances returned no instance")
        return str(instances[0]["InstanceId"])

    def _resolve_root_volume_size(self, ami_id: str) -> int:
        """The AMI's own root BDM size, mirroring OSWorld's resolver.

        OSWorld defaults to 40 GiB and grows to the AMI's root size
        (manager.py:105-140). The release AMI is 30 GiB, so the result is 40;
        the lookup exists so a future AMI larger than 40 does not fail to
        boot. The snapshot lookup is skipped: the BDM already carries the size.
        """

        default = 40
        return max(default, self._ami_root_volume_size(ami_id))

    def _ami_root_volume_size(self, ami_id: str) -> int:
        """The AMI's root BlockDeviceMapping size in GiB, with no floor applied.

        Split out from ``_resolve_root_volume_size`` because the two callers
        need different numbers from the same lookup. The default path wants
        OSWorld's 40 GiB floor; the override check wants the AMI's REAL size,
        since that -- not the floor -- is what AWS refuses to shrink below. One
        function returning max(40, ami) would accept a 35 GiB override against
        a 30 GiB AMI and reject one against a 45 GiB AMI with the wrong number
        in the message.

        Returns 0 when the image declares no EBS root mapping, which makes the
        floor check vacuous rather than wrong: an AMI whose size cannot be read
        must not manufacture a constraint the operator cannot satisfy.
        """

        response = self._clients.ec2.describe_images(ImageIds=[ami_id])
        images = response.get("Images", [])
        if not images:
            raise AllocationError(f"AMI {ami_id} is not visible in {self._region}")
        image = images[0]
        root = image.get("RootDeviceName")
        size = 0
        for mapping in image.get("BlockDeviceMappings", []):
            ebs = mapping.get("Ebs")
            if not ebs:
                continue
            if root is None or mapping.get("DeviceName") == root:
                size = max(size, int(ebs.get("VolumeSize", 0)))
                break
        return size

    def _refuse_volume_smaller_than_ami(self, ami_id: str, volume_gb: int) -> None:
        """Fail a root volume smaller than the AMI's snapshot, naming both sizes.

        EBS cannot restore a snapshot into a volume smaller than itself, so
        this launch is doomed; the only question is whether the operator finds
        out from a message naming the two numbers or from botocore's
        InvalidBlockDeviceMapping, which names neither.
        """

        ami_gb = self._ami_root_volume_size(ami_id)
        if ami_gb and volume_gb < ami_gb:
            raise AllocationError(
                f"root volume {volume_gb} GiB is smaller than AMI {ami_id}'s own "
                f"snapshot ({ami_gb} GiB); EBS cannot restore a snapshot into a "
                f"smaller volume, so raise AWS_ROOT_VOLUME_SIZE to at least {ami_gb}"
            )

    def _create_lease(self, instance_id: str, role_arn: str) -> None:
        fire_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl_seconds)
        self._clients.scheduler.create_schedule(
            Name=self._lease_ref,
            ScheduleExpression=f"at({fire_at.strftime('%Y-%m-%dT%H:%M:%S')})",
            FlexibleTimeWindow={"Mode": "OFF"},
            ActionAfterCompletion="DELETE",
            State="ENABLED",
            Description=f"lop osworld-v2 TTL terminate for {instance_id}",
            Target={
                "Arn": _TERMINATE_TARGET_ARN,
                "RoleArn": role_arn,
                "Input": json.dumps({"InstanceIds": [instance_id]}),
            },
        )

    def _terminate_best_effort(self, instance_id: str) -> None:
        # Used only on the lease-failure path inside allocate. Errors are
        # swallowed because the AllocationError about to be raised is the
        # signal; the runner's cleanup will re-terminate by tag and confirm.
        try:
            self._clients.ec2.terminate_instances(InstanceIds=[instance_id])
        except Exception:
            pass

    def _wait_ready(self, instance_id: str) -> str:
        """Wait for ``running``, then for the guest service to answer."""

        ec2 = self._clients.ec2
        waiter = ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 60})
        response = ec2.describe_instances(InstanceIds=[instance_id])
        found = _instances(response)
        if not found:
            raise AllocationError(f"instance {instance_id} vanished after running")
        public_ip = found[0].get("PublicIpAddress")
        if not public_ip:
            raise AllocationError(f"instance {instance_id} has no public IP")
        url = f"http://{public_ip}:{GUEST_PORT}{_READY_PATH}"
        deadline = time.monotonic() + self._readiness_timeout_s
        while True:
            try:
                if self._clients.http_get(url, 5.0) == 200:
                    return str(public_ip)
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise ReadinessTimeout(
                    f"guest {instance_id} at {public_ip} did not answer "
                    f"{_READY_PATH} within {self._readiness_timeout_s:.0f}s"
                )
            self._sleep(5.0)

    def _start_desktop_env(self, plan: ProvisioningPlan, task: TaskDescriptor) -> None:
        # Imports are deferred to here (and injected in tests): desktop_env
        # pulls gymnasium/pyautogui and reads the environment at import.
        if self._desktop_env_factory is not None:
            factory = self._desktop_env_factory
        else:
            from lop_osworld_v2_adapter import vendor_bridge

            factory = vendor_bridge.load_desktop_env()
        if self._task_factory is not None:
            task_instance = self._task_factory(task)
        else:
            from lop_osworld_v2_adapter import vendor_bridge

            task_instance = vendor_bridge.instantiate_task(task.module_name, task.task_id)
        # Upstream builds its own boto3 clients ambiently (``boto3.client(
        # 'ec2', region_name=...)`` in start_emulator/get_ip_address) with
        # NO credentials argument. The worker environment is stripped, so
        # those calls would fail with NoCredentialsError -- unless the
        # process default session carries the delivered values. This is the
        # one place credentials leave our Session object, and it is still
        # in-process memory, never the environment.
        self._install_default_session(plan.region)
        env = factory(
            provider_name="aws",
            region=plan.region,
            # path_to_vm=<instance id> makes DesktopEnv skip its own
            # allocation (desktop_env.py:110-113) and adopt ours.
            path_to_vm=self._instance_id,
            snapshot_name=plan.ami_id,
            # THE FIX for the rescue-wedge defect: upstream's default is
            # cache_dir="cache", resolved against the cwd — and the worker's
            # cwd IS the digest-pinned workspace. Routing the base under the
            # episode's owned root (absolute, outside the workspace) keeps
            # _download_setup / reset_cache_dir / getter downloads off the
            # pin, so the rescue worker's digest re-check still matches.
            cache_dir=str(self._cache_root),
            action_space="pyautogui",
            screen_size=plan.screen,
            headless=True,
            require_a11y_tree=False,
            os_type="Ubuntu",
            enable_proxy=plan.enable_proxy,
            client_password=plan.client_password,
            instance_type=plan.instance_type,
            use_public_ip=True,
        )
        self._seal_upstream(env)
        env.reset(task_config=task_instance)
        self._env = env

    def _install_default_session(self, region: str) -> None:
        if self._credentials is None:
            return
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - only reachable without the paid extra
            return
        boto3.setup_default_session(
            aws_access_key_id=self._credentials.access_key_id,
            aws_secret_access_key=self._credentials.secret_access_key,
            aws_session_token=self._credentials.session_token,
            region_name=region,
        )

    @staticmethod
    def _seal_upstream(env: Any) -> None:
        """Replace every upstream allocate/revert/release path with a raiser.

        Applied to the LIVE env object (not the class) immediately after
        construction and before ``reset``: ``DesktopEnv.__init__`` has
        already called ``provider.start_emulator``/``get_ip_address`` on our
        instance, which is the only upstream provider work we want. From
        here on, ``_revert_to_snapshot`` (a second ``reset`` once the env is
        used), ``provider.revert_to_snapshot`` (the untagged relaunch),
        ``manager.get_vm_path`` (upstream's own allocation), and
        ``close``/``stop_emulator`` (unconfirmed terminate) all raise
        ``UpstreamAllocationRefused`` BEFORE any boto3 call.

        The env is sealed at the instance level so the seal cannot leak
        into another ``DesktopEnv`` in the same process, and it survives
        upstream re-assigning ``self.provider``/``self.manager`` in
        ``_revert_to_snapshot`` only because that method is itself sealed
        first. Missing attributes are tolerated so an upstream refactor
        fails loudly at the seal's own test rather than silently here.
        """

        def refuse(name: str) -> Callable[..., Any]:
            def _refused(*_args: Any, **_kwargs: Any) -> Any:
                raise UpstreamAllocationRefused(
                    f"upstream DesktopEnv.{name} would allocate, revert, or release an "
                    "EC2 instance outside the adapter's tagged, leased, "
                    "rescue-resolvable path; refused before any boto3 call"
                )

            return _refused

        env._revert_to_snapshot = refuse("_revert_to_snapshot")
        env.close = refuse("close")
        env._save_state = refuse("_save_state")
        provider = getattr(env, "provider", None)
        if provider is not None:
            provider.revert_to_snapshot = refuse("provider.revert_to_snapshot")
            provider.stop_emulator = refuse("provider.stop_emulator")
            provider.save_state = refuse("provider.save_state")
        manager = getattr(env, "manager", None)
        if manager is not None:
            manager.get_vm_path = refuse("manager.get_vm_path")

    # ------------------------------------------------------------------
    # episode I/O
    # ------------------------------------------------------------------

    def _require_env(self) -> Any:
        if self._env is None:
            raise AllocationError("provider has no live DesktopEnv (allocate first)")
        return self._env

    async def observe(self) -> dict[str, Any]:
        env = self._require_env()
        raw = await asyncio.to_thread(env._get_obs)
        return dict(raw)

    async def execute(self, statements: list[str]) -> None:
        env = self._require_env()

        def run() -> None:
            for statement in statements:
                env.controller.execute_python_command(statement)
            # Let the desktop settle after the batch, exactly as OSWorld's
            # own ``step(pause=...)`` does. An empty batch still settles so a
            # pure WAIT advances the guest clock.
            self._sleep(self._action_delay_s)

        await asyncio.to_thread(run)

    async def evaluate(self) -> Any:
        env = self._require_env()

        def run() -> Any:
            capture = _JudgeErrorCapture()
            loggers = [logging.getLogger(name) for name in _JUDGE_LOGGERS]
            for logger in loggers:
                logger.addHandler(capture)
            try:
                raw = env.evaluate()
            finally:
                for logger in loggers:
                    logger.removeHandler(capture)
            if capture.messages:
                # A recorded ERROR means upstream swallowed an exception into
                # a score. Returning that score would seal a silent zero.
                raise scoring.ScoringUnavailable(
                    "judge/evaluator backend failed: " + "; ".join(capture.messages[:3])
                )
            return raw

        return await asyncio.to_thread(run)

    async def respond(self, prompt: str) -> str | None:
        env = self._env
        simulator = getattr(env, "user_simulator", None) if env is not None else None
        if simulator is None:
            return None
        answer = await asyncio.to_thread(simulator.respond, prompt)
        return str(answer)

    # ------------------------------------------------------------------
    # teardown (usable with only credentials + refs; see for_teardown)
    # ------------------------------------------------------------------

    async def terminate(self, instance_ref: str) -> str:
        return await asyncio.to_thread(self._terminate, instance_ref)

    def _terminate(self, instance_ref: str) -> str:
        episode_id = _episode_id_from_ref(instance_ref)
        ec2 = self._clients.ec2
        try:
            live = _instances(ec2.describe_instances(Filters=_tag_filters(episode_id)))
            if not live:
                return EVIDENCE_INSTANCE_ABSENT
            ids = [str(instance["InstanceId"]) for instance in live]
            ec2.terminate_instances(InstanceIds=ids)
        except Exception as error:
            if _error_code(error) == "UnauthorizedOperation":
                return EVIDENCE_TERMINATE_DENIED
            raise
        # Confirm. ``shutting-down`` is NOT confirmation: it has been
        # observed to linger, and the mapping in adapter._terminate_status
        # accepts only positive evidence of ``terminated``.
        deadline = time.monotonic() + self._terminate_timeout_s
        while True:
            response = ec2.describe_instances(InstanceIds=ids)
            states = {str(i.get("State", {}).get("Name")) for i in _instances(response)}
            if states and states <= {"terminated"}:
                return EVIDENCE_INSTANCE_TERMINATED
            if time.monotonic() >= deadline:
                return EVIDENCE_TERMINATE_UNCONFIRMED
            self._sleep(2.0)

    async def delete_schedule(self, lease_ref: str) -> str:
        return await asyncio.to_thread(self._delete_schedule, lease_ref)

    def _delete_schedule(self, lease_ref: str) -> str:
        try:
            self._clients.scheduler.delete_schedule(Name=lease_ref)
        except Exception as error:
            if _error_code(error) == "ResourceNotFoundException":
                return EVIDENCE_SCHEDULE_ABSENT
            return EVIDENCE_SCHEDULE_DELETE_FAILED
        return EVIDENCE_SCHEDULE_DELETED

    async def describe(self, instance_ref: str) -> dict[str, Any] | None:
        episode_id = _episode_id_from_ref(instance_ref)

        def run() -> dict[str, Any] | None:
            found = _instances(
                self._clients.ec2.describe_instances(Filters=_tag_filters(episode_id))
            )
            return found[0] if found else None

        return await asyncio.to_thread(run)

    # ------------------------------------------------------------------
    # audit (read-only, account-wide, no descriptor)
    # ------------------------------------------------------------------

    @staticmethod
    def audit(clients: _Clients) -> list[dict[str, Any]]:
        """Every live resource carrying the adapter tag, or ``[]``.

        The proof's assertion after every episode. Deliberately NOT a
        terminator: teardown happens only through a descriptor-driven rescue
        so every termination has a receipt. Instances, volumes and schedules
        are all listed so a leak of any kind shows.
        """

        found: list[dict[str, Any]] = []
        ec2 = clients.ec2
        for instance in _instances(ec2.describe_instances(Filters=_tag_filters(None))):
            tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
            found.append(
                {
                    "kind": "instance",
                    "id": instance.get("InstanceId"),
                    "state": instance.get("State", {}).get("Name"),
                    "episode": tags.get(EPISODE_TAG_KEY),
                    "launched_at": (
                        instance["LaunchTime"].isoformat()
                        if isinstance(instance.get("LaunchTime"), datetime)
                        else instance.get("LaunchTime")
                    ),
                }
            )
        volumes = ec2.describe_volumes(
            Filters=[{"Name": f"tag:{ADAPTER_TAG_KEY}", "Values": [ADAPTER_TAG_VALUE]}]
        )
        for volume in volumes.get("Volumes", []):
            tags = {t["Key"]: t["Value"] for t in volume.get("Tags", [])}
            found.append(
                {
                    "kind": "volume",
                    "id": volume.get("VolumeId"),
                    "state": volume.get("State"),
                    "episode": tags.get(EPISODE_TAG_KEY),
                }
            )
        schedules = clients.scheduler.list_schedules(NamePrefix="lop-ttl-")
        for schedule in schedules.get("Schedules", []):
            found.append(
                {
                    "kind": "schedule",
                    "id": schedule.get("Name"),
                    "state": schedule.get("State"),
                    "episode": str(schedule.get("Name", ""))[len("lop-ttl-") :] or None,
                }
            )
        return found

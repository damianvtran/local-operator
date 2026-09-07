"""AwsProvider against botocore's Stubber: the boto3 call shapes, in order.

No cloud calls are made; one control-boundary test uses loopback HTTP.
Every EC2 / Scheduler call is answered by a
``Stubber`` that ALSO asserts the request parameters, so these tests pin the
exact ``run_instances`` shape (ClientToken, both TagSpecifications, the
adapter tag the TTL role's condition requires), the schedule shape, and the
teardown confirmation loop. The failure paths are the point: a lease that
cannot be created must terminate the instance and raise; an instance that
lingers in ``shutting-down`` must report ``terminate-unconfirmed`` rather than
``instance-terminated``; a missing schedule must read as ``schedule-absent``.

The clock is virtual: ``sleep`` is injected and ``time.monotonic`` is patched
so the polling loops run in microseconds, and no assertion depends on wall
time (AGENTS.md "Timing, flakes").
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.stub import ANY, Stubber
from lop_osworld_v2_adapter import cleanup, provisioning, scoring, taskfile
from lop_osworld_v2_adapter.providers import aws as aws_mod
from lop_osworld_v2_adapter.providers.aws import (
    AllocationError,
    AwsCredentials,
    AwsProvider,
    GuestExecutionError,
    ReadinessTimeout,
    _Clients,
    ttl_seconds_for,
)

from local_operator import computer_input
from local_operator.evaluation.adapters.api import ScopedInfraValue
from tests.unit.evaluation.adapters.osworld import fixtures

REGION = "us-east-1"
EPISODE = "ep-aws-test"
# The episode-owned, absolute, workspace-external cache root the adapter mints
# in reset_start. Tests pass a fresh tmp path; allocate refuses without one.
_CACHE_ROOT_ARG = "cache_root"


@pytest.fixture(autouse=True)
def _hermetic_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """No ambient AWS configuration may reach these tests.

    The developer's shell may export ``AWS_PROFILE`` (a profile the CI runner
    does not have) or a default region; botocore reads both at client
    construction. Pointing the config and credentials files at empty paths
    and dropping the profile makes the tests hermetic AND pins the property
    the provider relies on: it is built from explicit values, not the
    environment.

    ``AWS_DEFAULT_PROFILE`` is in the list because botocore reads it as well
    as ``AWS_PROFILE``, and a docstring promising "no ambient AWS
    configuration" is simply wrong until it covers every variable the client
    reads: a developer with only that one exported got 24 ``ProfileNotFound``
    failures on a tree that is green on CI.
    """

    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_DEFAULT_REGION", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))


INSTANCE = "i-0123456789abcdef0"
CREDS = AwsCredentials(access_key_id="AKIATEST", secret_access_key="marker-secret")


def _infra() -> tuple[ScopedInfraValue, ...]:
    return tuple(
        ScopedInfraValue(name=name, purpose="benchmark_compute", value=value)
        for name, value in (
            ("AWS_REGION", REGION),
            ("AWS_SUBNET_ID", "subnet-1"),
            ("AWS_SECURITY_GROUP_ID", "sg-1"),
            ("AWS_SCHEDULER_ROLE_ARN", "arn:aws:iam::1:role/ttl"),
            ("OSWORLD_CLIENT_PASSWORD", "pw"),
            ("OSWORLD_FILE_BASE_URL", "https://assets"),
        )
    )


def _plan() -> provisioning.ProvisioningPlan:
    task = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/task_plain.py")
    return provisioning.resolve(task, episode_id=EPISODE, infra_values=_infra())


def _task() -> taskfile.TaskDescriptor:
    return taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/task_plain.py")


def _cache_root(tmp_path: Path) -> Path:
    """The episode-owned, absolute, workspace-external cache root the adapter
    mints in reset_start. Tests supply a fresh tmp path; allocate refuses
    without one because a missing root re-opens the digest-break defect."""

    root = tmp_path / "osworld-cache" / EPISODE
    root.mkdir(parents=True, exist_ok=True)
    return root


class _Stubs:
    """Two stubbed clients plus scripted HTTP probers and a virtual clock.

    ``guest_posts`` records every ``/execute`` body the provider sent, which is
    how the disk-preparation tests assert on what ran in the guest without a
    network. ``guest_responses`` scripts the replies: a mapping from a substring
    of the posted command to the body to return, so a test names only the step
    it cares about and every other step gets the benign default.
    """

    def __init__(self, *, http_codes: list[int] | None = None) -> None:
        self.ec2 = boto3.client(
            "ec2", region_name=REGION, aws_access_key_id="x", aws_secret_access_key="y"
        )
        self.scheduler = boto3.client(
            "scheduler", region_name=REGION, aws_access_key_id="x", aws_secret_access_key="y"
        )
        self.ec2_stub = Stubber(self.ec2)
        self.sched_stub = Stubber(self.scheduler)
        self.http_calls: list[str] = []
        self._http_codes = list(http_codes or [200])
        self.slept: list[float] = []
        self.now = 0.0

        def http_get(url: str, timeout: float) -> int:
            self.http_calls.append(url)
            return self._http_codes.pop(0) if len(self._http_codes) > 1 else self._http_codes[0]

        self.guest_posts: list[dict[str, Any]] = []
        self.guest_responses: list[tuple[str, dict[str, Any] | Exception]] = []
        # Enough free space that preparation is a no-op unless a test says
        # otherwise: every pre-existing allocate test wants the old behaviour.
        self.guest_default: dict[str, Any] | Exception = {
            "returncode": 0,
            "output": str(64 * 1024**3),
            "error": "",
        }

        def http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            self.guest_posts.append(payload)
            script = " ".join(payload.get("command", []))
            for needle, response in self.guest_responses:
                if needle in script:
                    if isinstance(response, Exception):
                        raise response
                    return response
            if isinstance(self.guest_default, Exception):
                raise self.guest_default
            return self.guest_default

        self.clients = _Clients(
            ec2=self.ec2,
            scheduler=self.scheduler,
            http_get=http_get,
            http_post_json=http_post_json,
        )

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now

    def __enter__(self) -> "_Stubs":
        self.ec2_stub.activate()
        self.sched_stub.activate()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ec2_stub.deactivate()
        self.sched_stub.deactivate()


def _provider(stubs: _Stubs, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> AwsProvider:
    monkeypatch.setattr(aws_mod.time, "monotonic", stubs.monotonic)
    # The instance_running waiter sleeps on its own; make that instant too.
    monkeypatch.setattr("botocore.waiter.time.sleep", lambda _s: None)
    return AwsProvider(
        CREDS,
        region=REGION,
        lease_ref=f"lop-ttl-{EPISODE}",
        ttl_seconds=3600,
        clients=stubs.clients,
        sleep=stubs.sleep,
        **kwargs,
    )


def _tags() -> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in _plan().tags]


def _expect_describe_images(stubs: _Stubs) -> None:
    stubs.ec2_stub.add_response(
        "describe_images",
        {
            "Images": [
                {
                    "ImageId": _plan().ami_id,
                    "RootDeviceName": "/dev/sda1",
                    "BlockDeviceMappings": [
                        {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 30}},
                    ],
                }
            ]
        },
        {"ImageIds": [_plan().ami_id]},
    )


def _expect_run_instances(
    stubs: _Stubs,
    plan: provisioning.ProvisioningPlan | None = None,
    *,
    volume_gb: int | None = None,
) -> None:
    # ``plan`` defaults to the standard one so every existing caller is
    # unchanged; a caller testing a resolved override passes its own.
    plan = plan if plan is not None else _plan()
    # ``volume_gb`` is separate from ``plan`` because the size AWS receives is
    # not always the plan's: with ``plan.volume_gb`` None the provider resolves
    # it from the AMI (40 = OSWorld's floor, which the AMI's 30 does not
    # lower). Defaulting to that keeps every existing caller unchanged while
    # letting an override test assert the size actually sent.
    expected_volume = volume_gb if volume_gb is not None else (plan.volume_gb or 40)
    stubs.ec2_stub.add_response(
        "run_instances",
        {"Instances": [{"InstanceId": INSTANCE}]},
        {
            "MaxCount": 1,
            "MinCount": 1,
            "ImageId": plan.ami_id,
            "InstanceType": plan.instance_type,
            "EbsOptimized": True,
            "InstanceInitiatedShutdownBehavior": "terminate",
            "ClientToken": plan.client_token,
            "NetworkInterfaces": [
                {
                    "DeviceIndex": 0,
                    "SubnetId": "subnet-1",
                    "AssociatePublicIpAddress": True,
                    "Groups": ["sg-1"],
                }
            ],
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": expected_volume,
                        "VolumeType": "gp3",
                        "Throughput": 1000,
                        "Iops": 4000,
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {"ResourceType": "instance", "Tags": _tags()},
                {"ResourceType": "volume", "Tags": _tags()},
            ],
        },
    )


def _expect_create_schedule(stubs: _Stubs) -> None:
    stubs.sched_stub.add_response(
        "create_schedule",
        {"ScheduleArn": "arn:aws:scheduler:us-east-1:1:schedule/default/x"},
        {
            "Name": f"lop-ttl-{EPISODE}",
            "ScheduleExpression": ANY,
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "ActionAfterCompletion": "DELETE",
            "State": "ENABLED",
            "Description": ANY,
            "Target": {
                "Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances",
                "RoleArn": "arn:aws:iam::1:role/ttl",
                "Input": json.dumps({"InstanceIds": [INSTANCE]}),
            },
        },
    )


def _instance(state: str, **extra: Any) -> dict[str, Any]:
    return {
        "Reservations": [
            {"Instances": [{"InstanceId": INSTANCE, "State": {"Name": state}, **extra}]}
        ]
    }


def _expect_running(stubs: _Stubs, public_ip: str = "203.0.113.5") -> None:
    # The waiter's describe, then the provider's own describe for the IP.
    stubs.ec2_stub.add_response(
        "describe_instances", _instance("running"), {"InstanceIds": [INSTANCE]}
    )
    stubs.ec2_stub.add_response(
        "describe_instances",
        _instance("running", PublicIpAddress=public_ip),
        {"InstanceIds": [INSTANCE]},
    )


class _UpstreamProviderShape:
    """Mirrors the upstream AWSProvider/AWSVMManager surface the seal covers.

    Each method records that upstream was reached and then calls
    ``boto3.client`` -- which the tests assert never happens, because the
    seal must raise BEFORE any boto3 call.
    """

    def __init__(self) -> None:
        self.reached: list[str] = []

    def _touch_boto3(self, name: str) -> None:
        self.reached.append(name)
        boto3.client("ec2", region_name=REGION)  # pragma: no cover - must be unreachable

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        self._touch_boto3("revert_to_snapshot")
        return "i-untagged"

    def stop_emulator(self, path_to_vm: str) -> None:
        self._touch_boto3("stop_emulator")

    def save_state(self, path_to_vm: str, snapshot_name: str) -> None:
        self._touch_boto3("save_state")

    def get_vm_path(self, **kwargs: Any) -> str:
        self._touch_boto3("get_vm_path")
        return "i-untagged"


class _FakeEnv:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls: list[Any] = []
        self.user_simulator = None
        self.controller = self
        self.pkgs_prefix = "import pyautogui; {command}"
        self.provider = _UpstreamProviderShape()
        self.manager = self.provider
        self.is_environment_used = False
        self.path_to_vm: str = str(kwargs.get("path_to_vm", ""))
        self.snapshot_name: str = str(kwargs.get("snapshot_name", ""))

    def _revert_to_snapshot(self) -> None:
        # Upstream desktop_env.py:202-212, condensed.
        self.path_to_vm = self.provider.revert_to_snapshot(self.path_to_vm, self.snapshot_name)

    def _save_state(self, snapshot_name: str = "") -> None:
        self.provider.save_state(self.path_to_vm, snapshot_name)

    def reset(self, task_config: Any) -> None:
        # Upstream desktop_env.py:329-336: a used env reverts before setup.
        if self.is_environment_used:
            self._revert_to_snapshot()
        self.is_environment_used = True
        self.reset_calls.append(task_config)

    def _get_obs(self) -> dict[str, Any]:
        return {"screenshot": b"png", "instruction": "do it"}

    def execute_python_command(self, command: str) -> None:
        self.kwargs.setdefault("executed", []).append(command)

    def evaluate(self) -> Any:
        return 0.5

    def close(self) -> None:
        # Upstream desktop_env.py:282-284.
        self.provider.stop_emulator(self.path_to_vm)


# ---------------------------------------------------------------------------
# allocate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allocate_launches_with_client_token_and_both_tag_specs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _Stubs() as stubs:
        _expect_describe_images(stubs)
        _expect_run_instances(stubs)
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        envs: list[_FakeEnv] = []

        def factory(**kwargs: Any) -> _FakeEnv:
            env = _FakeEnv(**kwargs)
            envs.append(env)
            return env

        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=factory,
            task_factory=lambda t: {"id": t.task_id},
        )
        await provider.allocate(_plan(), _task(), cache_root=_cache_root(tmp_path))
        stubs.ec2_stub.assert_no_pending_responses()
        stubs.sched_stub.assert_no_pending_responses()

    # DesktopEnv adopted OUR instance and never allocated its own.
    assert len(envs) == 1
    assert envs[0].kwargs["path_to_vm"] == INSTANCE
    assert envs[0].kwargs["provider_name"] == "aws"
    assert envs[0].kwargs["region"] == REGION
    assert envs[0].kwargs["use_public_ip"] is True
    assert envs[0].reset_calls == [{"id": "task_plain"}]
    # Readiness probed the guest's control port on the public IP.
    assert stubs.http_calls == ["http://203.0.113.5:5000/terminal"]


@pytest.mark.asyncio
async def test_an_instance_type_override_reaches_the_real_run_instances_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The override must survive resolve -> plan -> EC2, not just resolve.

    A unit test of the pure function would still pass if the plan field were
    dropped on the way to ``run_instances``, which is precisely the failure
    that would leave the operator on the starved burstable box while every
    test stayed green. The stubbed client asserts the exact launch parameters,
    so ``InstanceType`` here is the value AWS would actually receive.
    """

    task = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/task_plain.py")
    infra = _infra() + (
        ScopedInfraValue(name="AWS_INSTANCE_TYPE", purpose="benchmark_compute", value="m5.xlarge"),
    )
    plan = provisioning.resolve(task, episode_id=EPISODE, infra_values=infra)
    assert plan.instance_type == "m5.xlarge"

    with _Stubs() as stubs:
        _expect_describe_images(stubs)
        # Built from the OVERRIDDEN plan: the stub rejects the call if any
        # launch parameter differs, so a dropped override fails right here.
        _expect_run_instances(stubs, plan=plan)
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=_FakeEnv,
            task_factory=lambda t: {"id": t.task_id},
        )
        await provider.allocate(plan, task, cache_root=_cache_root(tmp_path))
        stubs.ec2_stub.assert_no_pending_responses()


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", [False, True])
@pytest.mark.parametrize("policy", [None, "false", "true"])
async def test_proxy_policy_reaches_desktop_env_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hint: bool, policy: str | None
) -> None:
    task = taskfile.TaskDescriptor(
        task_id="synthetic", instruction="", source_sha256="0" * 64, proxy=hint
    )
    infra = _infra() + (
        ()
        if policy is None
        else (
            ScopedInfraValue(
                name="OSWORLD_ENABLE_PROXY", purpose="benchmark_compute", value=policy
            ),
        )
    )
    plan = provisioning.resolve(task, episode_id=EPISODE, infra_values=infra)
    captured: list[_FakeEnv] = []

    def factory(**kwargs: Any) -> _FakeEnv:
        env = _FakeEnv(**kwargs)
        captured.append(env)
        return env

    with _Stubs() as stubs:
        _expect_describe_images(stubs)
        _expect_run_instances(stubs, plan=plan)
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=factory,
            task_factory=lambda t: {"id": t.task_id, "proxy": t.proxy},
        )
        await provider.allocate(plan, task, cache_root=_cache_root(tmp_path))
        stubs.ec2_stub.assert_no_pending_responses()
    assert len(captured) == 1
    assert captured[0].kwargs["enable_proxy"] is (hint if policy is None else policy == "true")


@pytest.mark.asyncio
async def test_desktop_env_cache_dir_is_absolute_and_outside_the_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE regression: upstream's default cache_dir="cache" is cwd-relative and
    the worker's cwd is the pinned workspace, so the paid episode's
    _download_setup broke the rescue digest. The adapter must hand DesktopEnv
    the episode's absolute cache root instead."""

    with _Stubs() as stubs:
        _expect_describe_images(stubs)
        _expect_run_instances(stubs)
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        envs: list[_FakeEnv] = []

        def factory(**kwargs: Any) -> _FakeEnv:
            env = _FakeEnv(**kwargs)
            envs.append(env)
            return env

        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=factory,
            task_factory=lambda t: {"id": t.task_id},
        )
        cache_root = _cache_root(tmp_path)
        await provider.allocate(_plan(), _task(), cache_root=cache_root)

    cache_dir = Path(envs[0].kwargs["cache_dir"])
    assert cache_dir.is_absolute(), "cache_dir must never be cwd-relative"
    assert cache_dir == cache_root
    # It must not be the workspace the worker runs in (the cwd here).
    cwd = Path.cwd().resolve()
    assert cwd not in cache_dir.parents and cache_dir != cwd
    # And never under a volatile /tmp that macOS purges mid-run.
    assert not str(cache_dir).startswith(("/tmp", "/private/tmp")) or str(cache_dir).startswith(
        str(tmp_path)
    )


@pytest.mark.asyncio
async def test_allocate_refuses_without_a_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing cache root must fail loudly before any boto3 call: defaulting
    to cwd would silently re-open the digest-break defect.

    ``cache_root`` is typed as a required ``Path``, so a caller that forgets it
    is caught at type-check time; the ``type: ignore`` below is deliberate —
    it exercises the RUNTIME guard that still has to hold for a caller passing
    ``None`` dynamically."""

    with _Stubs() as stubs:
        provider = _provider(stubs, monkeypatch, desktop_env_factory=_FakeEnv)
        with pytest.raises(AllocationError, match="cache root"):
            await provider.allocate(_plan(), _task(), cache_root=None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Guest disk preparation (see ``guest_disk`` for the measurements)
# ----------------------------------------------------------------------


async def _allocate_with_guest(
    stubs: _Stubs, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Drive a full allocate and return the episode cache root it wrote into."""

    _expect_describe_images(stubs)
    _expect_run_instances(stubs)
    _expect_create_schedule(stubs)
    _expect_running(stubs)
    provider = _provider(
        stubs,
        monkeypatch,
        desktop_env_factory=_FakeEnv,
        task_factory=lambda t: {"id": t.task_id},
    )
    root = _cache_root(tmp_path)
    await provider.allocate(_plan(), _task(), cache_root=root)
    return root


class _ResetRecordingEnv(_FakeEnv):
    """A ``_FakeEnv`` that records how many guest posts preceded its reset.

    Ordering is only assertable against a shared timeline, so the env samples
    the guest's post log at the moment upstream's ``reset`` runs. Comparing that
    count with the final one is what distinguishes "prepared the disk, then
    reset" from "reset, then prepared" -- two orderings that otherwise produce
    identical reports and identical call sets.
    """

    posts_at_reset: int | None = None
    stubs: _Stubs | None = None

    def reset(self, task_config: Any) -> None:
        assert _ResetRecordingEnv.stubs is not None
        _ResetRecordingEnv.posts_at_reset = len(_ResetRecordingEnv.stubs.guest_posts)
        super().reset(task_config)


@pytest.mark.asyncio
async def test_allocate_reclaims_a_tight_guest_before_upstream_resets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The disk is reclaimed BEFORE the first observation, and it is recorded.

    Ordering is the whole point: the released AMI ships ~93% full and its own
    snapd fills the rest on a clock (100% used / 0 bytes free at t+383s, first
    ``ObservationPhaseError`` at t+424s), so hygiene that ran after upstream's
    reset would already be too late for the frame that reset captures.
    """

    _ResetRecordingEnv.posts_at_reset = None
    with _Stubs() as stubs:
        _ResetRecordingEnv.stubs = stubs
        stubs.guest_default = {"returncode": 0, "output": str(2_200_000_000), "error": ""}
        _expect_describe_images(stubs)
        _expect_run_instances(stubs)
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=_ResetRecordingEnv,
            task_factory=lambda t: {"id": t.task_id},
        )
        root = _cache_root(tmp_path)
        await provider.allocate(_plan(), _task(), cache_root=root)

    scripts = [" ".join(post["command"]) for post in stubs.guest_posts]
    assert any("snap refresh --hold=forever" in script for script in scripts)
    assert any("/var/lib/snapd/cache" in script for script in scripts)

    # EVERY hygiene post landed before upstream's reset captured the first
    # frame. Preparation that ran afterwards would leave that frame -- and the
    # setup writes preceding it -- on the disk this step exists to protect.
    posts_at_reset = _ResetRecordingEnv.posts_at_reset
    assert posts_at_reset is not None, "upstream reset never ran"
    assert posts_at_reset == len(stubs.guest_posts)
    assert posts_at_reset > 0
    # Every post uses the endpoint contract upstream itself uses: an argv list
    # with ``shell: false``, so the server execs it directly.
    assert all(post["shell"] is False for post in stubs.guest_posts)
    assert all(isinstance(post["command"], list) for post in stubs.guest_posts)

    report = json.loads((root / "guest-preparation.json").read_bytes())
    assert report["reclamation_attempted"] is True
    assert report["free_bytes_before"] == 2_200_000_000


@pytest.mark.asyncio
async def test_allocate_leaves_a_roomy_guest_alone_but_still_records_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _Stubs() as stubs:
        root = await _allocate_with_guest(stubs, monkeypatch, tmp_path)

    scripts = [" ".join(post["command"]) for post in stubs.guest_posts]
    assert not any("snap refresh" in script for script in scripts)
    report = json.loads((root / "guest-preparation.json").read_bytes())
    assert report["reclamation_attempted"] is False
    assert report["reason"] == "above-threshold"


@pytest.mark.asyncio
async def test_an_unreachable_guest_control_server_does_not_fail_the_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FAIL SOFT, at the provider boundary this time.

    The readiness probe already passed, so the guest is up; a control server
    that then refuses the hygiene posts must cost the episode nothing. An
    allocation that raised here would destroy an episode that would have run.
    """

    with _Stubs() as stubs:
        stubs.guest_default = ConnectionError("connection refused")
        root = await _allocate_with_guest(stubs, monkeypatch, tmp_path)
        stubs.ec2_stub.assert_no_pending_responses()
        stubs.sched_stub.assert_no_pending_responses()

    report = json.loads((root / "guest-preparation.json").read_bytes())
    assert report["free_bytes_before"] is None
    assert all(step["status"] == "unreachable" for step in report["steps"])


@pytest.mark.asyncio
async def test_a_guest_returning_a_body_without_a_returncode_reads_as_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A server that did not say the command succeeded did not say it succeeded.

    Defaulting a missing ``returncode`` to 0 would let a malformed reply be
    recorded as a successful reclamation, which is the one thing the evidence
    must not be able to claim falsely.
    """

    with _Stubs() as stubs:
        stubs.guest_default = {"output": "", "error": "no returncode field"}
        root = await _allocate_with_guest(stubs, monkeypatch, tmp_path)

    report = json.loads((root / "guest-preparation.json").read_bytes())
    assert report["free_bytes_before"] is None
    assert all(step["status"] == "failed" for step in report["steps"])


@pytest.mark.asyncio
async def test_the_preparation_report_never_carries_the_client_password(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The password reaches the guest through ``sudo -S`` and stops there.

    Same discipline as every other secret on this boundary: it may cross the
    wire to the guest, and it may not be written to the operator's disk.
    """

    with _Stubs() as stubs:
        stubs.guest_default = {"returncode": 0, "output": str(2_200_000_000), "error": ""}
        root = await _allocate_with_guest(stubs, monkeypatch, tmp_path)

    # ``_plan()``'s OSWORLD_CLIENT_PASSWORD is "pw" -- too short to search for
    # without false positives, so assert on the pipeline shape that carries it.
    scripts = [" ".join(post["command"]) for post in stubs.guest_posts]
    assert any("| sudo -S" in script for script in scripts)
    report = (root / "guest-preparation.json").read_text()
    assert "sudo -S" not in report
    assert "snap refresh" not in report


@pytest.mark.asyncio
async def test_schedule_is_created_before_readiness_and_its_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lease covers the boot window: it is created right after
    run_instances, and if it cannot be, the instance is terminated and
    allocate raises. No warning path exists."""

    with _Stubs() as stubs:
        _expect_describe_images(stubs)
        _expect_run_instances(stubs)
        stubs.sched_stub.add_client_error(
            "create_schedule",
            service_error_code="ValidationException",
            service_message="role cannot be assumed",
        )
        stubs.ec2_stub.add_response(
            "terminate_instances",
            {"TerminatingInstances": [{"InstanceId": INSTANCE}]},
            {"InstanceIds": [INSTANCE]},
        )
        provider = _provider(stubs, monkeypatch, desktop_env_factory=_FakeEnv)
        with pytest.raises(AllocationError, match="TTL lease .* could not be created"):
            await provider.allocate(_plan(), _task(), cache_root=_cache_root(tmp_path))
        stubs.ec2_stub.assert_no_pending_responses()
        stubs.sched_stub.assert_no_pending_responses()
    # No readiness wait happened: the prober was never called.
    assert stubs.http_calls == []


@pytest.mark.asyncio
async def test_readiness_timeout_raises_after_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _Stubs(http_codes=[503]) as stubs:
        _expect_describe_images(stubs)
        _expect_run_instances(stubs)
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        provider = _provider(
            stubs, monkeypatch, desktop_env_factory=_FakeEnv, readiness_timeout_s=12.0
        )
        with pytest.raises(ReadinessTimeout):
            await provider.allocate(_plan(), _task(), cache_root=_cache_root(tmp_path))
    # 12 s deadline at 5 s per probe: probes at t=0, 5, 10, 15; the last one
    # is past the deadline, so it raises instead of sleeping again.
    assert len(stubs.http_calls) == 4
    assert stubs.slept == [5.0, 5.0, 5.0]


@pytest.mark.asyncio
async def test_allocate_refuses_a_plan_for_another_region(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region="eu-west-1", lease_ref="lop-ttl-x", clients=stubs.clients
        )
        with pytest.raises(AllocationError, match="region"):
            await provider.allocate(_plan(), _task(), cache_root=_cache_root(tmp_path))


# ---------------------------------------------------------------------------
# the upstream boundary is sealed: no second launch, no unconfirmed close
# ---------------------------------------------------------------------------


async def _allocated(
    stubs: _Stubs, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[AwsProvider, _FakeEnv]:
    _expect_describe_images(stubs)
    _expect_run_instances(stubs)
    _expect_create_schedule(stubs)
    _expect_running(stubs)
    envs: list[_FakeEnv] = []

    def factory(**kwargs: Any) -> _FakeEnv:
        env = _FakeEnv(**kwargs)
        envs.append(env)
        return env

    provider = _provider(
        stubs, monkeypatch, desktop_env_factory=factory, task_factory=lambda t: {"id": t.task_id}
    )
    await provider.allocate(_plan(), _task(), cache_root=_cache_root(tmp_path))
    stubs.ec2_stub.assert_no_pending_responses()
    return provider, envs[0]


@pytest.mark.asyncio
async def test_a_second_reset_on_a_used_env_is_refused_before_any_boto3_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MAJOR-1. Upstream's second ``reset`` routes through
    ``_revert_to_snapshot`` -> ``AWSProvider.revert_to_snapshot``, which
    ``run_instances`` a NEW instance with no ClientToken, no lop:adapter tag
    and no TTL -- invisible to the audit, unreachable by rescue. The seal must
    raise before that provider method runs at all.

    The Stubber has no pending responses at this point, so any boto3 call
    would ALSO fail loudly -- but the assertion is on ``reached`` so the test
    proves the refusal happened before upstream code, not merely that boto3
    was unhappy afterwards.
    """

    with _Stubs() as stubs:
        provider, env = await _allocated(stubs, monkeypatch, tmp_path)
        assert env.reset_calls == [{"id": "task_plain"}]
        assert env.is_environment_used is True
        with pytest.raises(aws_mod.UpstreamAllocationRefused, match="_revert_to_snapshot"):
            env.reset(task_config={"id": "again"})
        assert env.provider.reached == []
        assert env.reset_calls == [{"id": "task_plain"}]
        # The direct provider paths are sealed too, not just the DesktopEnv entry.
        with pytest.raises(aws_mod.UpstreamAllocationRefused, match="revert_to_snapshot"):
            env.provider.revert_to_snapshot("i-x", "ami-x")
        with pytest.raises(aws_mod.UpstreamAllocationRefused, match="get_vm_path"):
            env.manager.get_vm_path()
        with pytest.raises(aws_mod.UpstreamAllocationRefused, match="_save_state"):
            env._save_state("snap")
        assert env.provider.reached == []


@pytest.mark.asyncio
async def test_upstream_close_is_refused_so_teardown_is_only_ever_confirmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _Stubs() as stubs:
        provider, env = await _allocated(stubs, monkeypatch, tmp_path)
        with pytest.raises(aws_mod.UpstreamAllocationRefused, match="close"):
            env.close()
        with pytest.raises(aws_mod.UpstreamAllocationRefused, match="stop_emulator"):
            env.provider.stop_emulator("i-x")
        assert env.provider.reached == []


def test_the_seal_covers_every_upstream_launch_and_release_method() -> None:
    """Pins the seal against the PINNED upstream: if a future OSWorld adds
    another path to ``run_instances``/``terminate_instances`` on the
    provider or manager, this list is where it must be added, and the
    static scan of the real checkout below is what flags it."""

    import os
    import pwd
    import re
    from pathlib import Path

    # The suite's conftest points HOME at a scratch directory; the pinned
    # checkout lives in the REAL home (or wherever OSWORLD_INPUTS_ROOT says).
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    inputs_root = Path(os.environ.get("OSWORLD_INPUTS_ROOT", real_home / "worktrees" / "osworld"))
    prepared = inputs_root / "prepared"
    provider_src = prepared / "desktop_env" / "providers" / "aws" / "provider.py"
    manager_src = prepared / "desktop_env" / "providers" / "aws" / "manager.py"
    if not provider_src.exists():  # pragma: no cover - inputs root absent on CI
        pytest.skip("pinned OSWorld checkout not present")
    sealed = {
        "revert_to_snapshot",
        "stop_emulator",
        "save_state",
        "get_vm_path",
    }
    launching_calls = re.compile(r"run_instances|terminate_instances|create_image")

    def launching_functions(source: str) -> set[str]:
        """Every function whose body reaches a launch call, DIRECTLY or through
        another function in the same module (``get_vm_path`` -> ``_allocate_vm``).
        AST-based: a regex on ``def name(self...)`` skipped signatures with a
        nested paren (``screen_size=(1920, 1080)``) and module-level helpers."""

        import ast

        tree = ast.parse(source)
        bodies: dict[str, str] = {}
        calls: dict[str, set[str]] = {}
        # Only module-level functions and class methods are reachable on the
        # env; a nested def (``_allocate_vm``'s ``signal_handler``) is
        # covered by sealing whatever encloses it.
        scopes: list[ast.Module | ast.ClassDef] = [tree]
        scopes.extend(node for node in tree.body if isinstance(node, ast.ClassDef))
        for scope in scopes:
            for node in scope.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    bodies[node.name] = ast.unparse(node)
                    calls[node.name] = {
                        c.func.id
                        for c in ast.walk(node)
                        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    }
        direct = {name for name, body in bodies.items() if launching_calls.search(body)}
        # Transitive closure over same-module calls.
        found = set(direct)
        changed = True
        while changed:
            changed = False
            for name, callees in calls.items():
                if name not in found and callees & found:
                    found.add(name)
                    changed = True
        return found

    launching: set[str] = set()
    for src in (provider_src, manager_src):
        launching |= launching_functions(src.read_text())
    # Module-level helpers are not reachable on the env object, so they are
    # covered by sealing their callers; assert they were found at all so the
    # scan is provably looking at manager.py.
    assert "_allocate_vm" in launching
    assert "get_vm_path" in launching
    methods = launching - {"_allocate_vm"}
    # start_emulator only start_instances/describe -- never launches -- and is
    # exactly the one upstream method we rely on during __init__.
    assert methods <= sealed, f"unsealed upstream launch/release methods: {methods - sealed}"


def test_the_judge_capture_file_set_matches_the_pinned_upstream() -> None:
    """MINOR-4 (round 2). The capture filters on basename; a rename upstream
    would silently stop it. Pin the set against the pinned checkout."""

    import os
    import pwd
    from pathlib import Path

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    inputs_root = Path(os.environ.get("OSWORLD_INPUTS_ROOT", real_home / "worktrees" / "osworld"))
    evaluators = inputs_root / "prepared" / "desktop_env" / "evaluators"
    if not evaluators.exists():  # pragma: no cover - inputs root absent on CI
        pytest.skip("pinned OSWorld checkout not present")
    present = {p.name for p in evaluators.rglob("*.py")}
    assert aws_mod._JUDGE_SOURCE_FILES <= present, aws_mod._JUDGE_SOURCE_FILES - present
    # And those files really log on the loggers we attach to.
    for name in aws_mod._JUDGE_SOURCE_FILES:
        source = next(evaluators.rglob(name)).read_text()
        assert any(f'getLogger("{logger}")' in source for logger in aws_mod._JUDGE_LOGGERS), name


# ---------------------------------------------------------------------------
# terminate / delete_schedule / describe
# ---------------------------------------------------------------------------


def _tag_filters() -> list[dict[str, Any]]:
    return [
        {"Name": "tag:lop:episode", "Values": [EPISODE]},
        {"Name": "tag:lop:adapter", "Values": ["osworld-v2"]},
        {"Name": "instance-state-name", "Values": list(aws_mod.LIVE_STATES)},
    ]


@pytest.mark.asyncio
async def test_terminate_polls_until_terminated(monkeypatch: pytest.MonkeyPatch) -> None:
    with _Stubs() as stubs:
        stubs.ec2_stub.add_response(
            "describe_instances", _instance("running"), {"Filters": _tag_filters()}
        )
        stubs.ec2_stub.add_response(
            "terminate_instances",
            {"TerminatingInstances": [{"InstanceId": INSTANCE}]},
            {"InstanceIds": [INSTANCE]},
        )
        stubs.ec2_stub.add_response(
            "describe_instances", _instance("shutting-down"), {"InstanceIds": [INSTANCE]}
        )
        stubs.ec2_stub.add_response(
            "describe_instances", _instance("terminated"), {"InstanceIds": [INSTANCE]}
        )
        provider = AwsProvider.for_teardown(
            CREDS, region=REGION, lease_ref=f"lop-ttl-{EPISODE}", clients=stubs.clients
        )
        provider._sleep = stubs.sleep
        monkeypatch.setattr(aws_mod.time, "monotonic", stubs.monotonic)
        code = await provider.terminate(f"lop-ep-{EPISODE}")
        stubs.ec2_stub.assert_no_pending_responses()
    assert code == cleanup.EVIDENCE_INSTANCE_TERMINATED
    assert stubs.slept == [2.0]


@pytest.mark.asyncio
async def test_shutting_down_at_deadline_is_unconfirmed_not_terminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shutting-down`` lingers in practice. It must NOT be reported as
    confirmed: the adapter maps only ``instance-terminated`` to succeeded."""

    with _Stubs() as stubs:
        stubs.ec2_stub.add_response(
            "describe_instances", _instance("running"), {"Filters": _tag_filters()}
        )
        stubs.ec2_stub.add_response(
            "terminate_instances",
            {"TerminatingInstances": [{"InstanceId": INSTANCE}]},
            {"InstanceIds": [INSTANCE]},
        )
        for _ in range(4):
            stubs.ec2_stub.add_response(
                "describe_instances", _instance("shutting-down"), {"InstanceIds": [INSTANCE]}
            )
        provider = AwsProvider.for_teardown(
            CREDS, region=REGION, lease_ref=f"lop-ttl-{EPISODE}", clients=stubs.clients
        )
        provider._sleep = stubs.sleep
        provider._terminate_timeout_s = 6.0
        monkeypatch.setattr(aws_mod.time, "monotonic", stubs.monotonic)
        code = await provider.terminate(f"lop-ep-{EPISODE}")
        stubs.ec2_stub.assert_no_pending_responses()
    assert code == cleanup.EVIDENCE_TERMINATE_UNCONFIRMED


@pytest.mark.asyncio
async def test_no_tagged_instance_is_absent() -> None:
    with _Stubs() as stubs:
        stubs.ec2_stub.add_response(
            "describe_instances", {"Reservations": []}, {"Filters": _tag_filters()}
        )
        provider = AwsProvider.for_teardown(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients
        )
        assert await provider.terminate(f"lop-ep-{EPISODE}") == cleanup.EVIDENCE_INSTANCE_ABSENT
        stubs.ec2_stub.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_unauthorized_terminate_is_denied() -> None:
    with _Stubs() as stubs:
        stubs.ec2_stub.add_response(
            "describe_instances", _instance("running"), {"Filters": _tag_filters()}
        )
        stubs.ec2_stub.add_client_error(
            "terminate_instances", service_error_code="UnauthorizedOperation"
        )
        provider = AwsProvider.for_teardown(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients
        )
        assert await provider.terminate(f"lop-ep-{EPISODE}") == cleanup.EVIDENCE_TERMINATE_DENIED


@pytest.mark.asyncio
async def test_terminate_rejects_a_ref_that_is_not_a_tag_ref() -> None:
    with _Stubs() as stubs:
        provider = AwsProvider.for_teardown(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients
        )
        with pytest.raises(ValueError, match="lop-ep-"):
            await provider.terminate("i-0123456789abcdef0")


@pytest.mark.asyncio
async def test_delete_schedule_codes() -> None:
    with _Stubs() as stubs:
        stubs.sched_stub.add_response("delete_schedule", {}, {"Name": "lop-ttl-a"})
        stubs.sched_stub.add_client_error(
            "delete_schedule", service_error_code="ResourceNotFoundException"
        )
        stubs.sched_stub.add_client_error(
            "delete_schedule", service_error_code="ThrottlingException"
        )
        provider = AwsProvider.for_teardown(
            CREDS, region=REGION, lease_ref="lop-ttl-a", clients=stubs.clients
        )
        assert await provider.delete_schedule("lop-ttl-a") == cleanup.EVIDENCE_SCHEDULE_DELETED
        assert await provider.delete_schedule("lop-ttl-a") == cleanup.EVIDENCE_SCHEDULE_ABSENT
        assert (
            await provider.delete_schedule("lop-ttl-a") == cleanup.EVIDENCE_SCHEDULE_DELETE_FAILED
        )
        stubs.sched_stub.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_describe_resolves_by_tag() -> None:
    with _Stubs() as stubs:
        stubs.ec2_stub.add_response(
            "describe_instances", _instance("running"), {"Filters": _tag_filters()}
        )
        stubs.ec2_stub.add_response(
            "describe_instances", {"Reservations": []}, {"Filters": _tag_filters()}
        )
        provider = AwsProvider.for_teardown(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients
        )
        found = await provider.describe(f"lop-ep-{EPISODE}")
        assert found is not None and found["InstanceId"] == INSTANCE
        assert await provider.describe(f"lop-ep-{EPISODE}") is None


# ---------------------------------------------------------------------------
# evaluate: a judge error becomes ScoringUnavailable, never a zero
# ---------------------------------------------------------------------------


def _log_from(logger_name: str, pathname: str, message: str) -> None:
    """Emit an ERROR record attributed to ``pathname``, as if from that file.

    ``LogRecord.pathname`` is what the capture filters on; ``makeRecord``
    lets the test set it to the upstream file that would really have logged.
    """

    logger = logging.getLogger(logger_name)
    record = logger.makeRecord(logger_name, logging.ERROR, pathname, 1, message, (), None)
    logger.handle(record)


_LLM_METRICS = "/site-packages/desktop_env/evaluators/metrics/llm_metrics.py"
_DESKTOP_ENV = "/site-packages/desktop_env/desktop_env.py"


@pytest.mark.asyncio
async def test_a_benign_setup_retry_error_does_not_mask_a_good_score() -> None:
    """MAJOR-3. ``desktopenv.env`` carries both the judge's swallowed errors
    (llm_metrics.py logs on it) and benign setup-retry ERRORs from
    desktop_env.py. A good score after a recoverable retry must be SCORED."""

    class _RetriedThenScoredEnv(_FakeEnv):
        def evaluate(self) -> Any:
            _log_from(
                "desktopenv.env",
                _DESKTOP_ENV,
                "Environment setup failed, retrying (1/5)...",
            )
            _log_from("desktopenv.env", _DESKTOP_ENV, "No evaluator configured for task x")
            return 0.75

    with _Stubs() as stubs:
        provider = AwsProvider(CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients)
        provider._env = _RetriedThenScoredEnv()
        assert await provider.evaluate() == 0.75


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("logger_name", "pathname", "message"),
    [
        ("desktopenv.env", _LLM_METRICS, "Error in compare_text_with_llm: boom"),
        ("desktopenv.env", _LLM_METRICS, "Error in compare_images_with_llm: 401"),
        ("desktopenv.eval_model", "/x/desktop_env/evaluators/model_client.py", "no key"),
    ],
)
async def test_a_judge_backend_error_raises_scoring_unavailable(
    logger_name: str, pathname: str, message: str
) -> None:
    """The other direction: an ERROR from a judge-owned file, on ANY of the
    loggers those files use, turns the swallowed 0.0 into ScoringUnavailable."""

    class _JudgeFailsEnv(_FakeEnv):
        def evaluate(self) -> Any:
            _log_from(logger_name, pathname, message)
            return 0.0

    with _Stubs() as stubs:
        provider = AwsProvider(CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients)
        provider._env = _JudgeFailsEnv()
        with pytest.raises(scoring.ScoringUnavailable, match="judge"):
            await provider.evaluate()


@pytest.mark.asyncio
async def test_judge_error_and_benign_error_together_still_raise() -> None:
    class _BothEnv(_FakeEnv):
        def evaluate(self) -> Any:
            _log_from("desktopenv.env", _DESKTOP_ENV, "Environment setup failed, retrying (2/5)...")
            _log_from("desktopenv.env", _LLM_METRICS, "Error in _compare_answers_with_llm: timeout")
            return 0.0

    with _Stubs() as stubs:
        provider = AwsProvider(CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients)
        provider._env = _BothEnv()
        with pytest.raises(scoring.ScoringUnavailable, match="_compare_answers_with_llm"):
            await provider.evaluate()


@pytest.mark.asyncio
async def test_evaluate_returns_raw_when_the_judge_is_quiet() -> None:
    with _Stubs() as stubs:
        provider = AwsProvider(CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients)
        provider._env = _FakeEnv()
        assert await provider.evaluate() == 0.5
        # The capture handler was removed afterwards, on every logger.
        for name in aws_mod._JUDGE_LOGGERS:
            assert not any(
                isinstance(h, aws_mod._JudgeErrorCapture) for h in logging.getLogger(name).handlers
            )


@pytest.mark.asyncio
async def test_execute_over_real_http_reports_partial_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the control boundary, not X11: no guest display exists on CI."""
    import subprocess
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from lop_osworld_v2_adapter.providers import aws

    marker = tmp_path / "committed.txt"
    requests_seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests_seen.append(payload)
            assert self.path == "/execute"
            assert payload["shell"] is False
            # The guest resolves `python` in its own environment. Use this test
            # environment's interpreter, but execute the received source intact.
            completed = subprocess.run(
                [sys.executable, *payload["command"][1:]], capture_output=True, text=True
            )
            body = json.dumps(
                {
                    "returncode": completed.returncode,
                    "output": completed.stdout,
                    "error": completed.stderr,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(aws, "GUEST_PORT", server.server_port)
    try:
        with _Stubs() as stubs:
            # The real HTTP closure makes requests; boto clients are constructed
            # with fixture credentials and are never invoked by execute().
            stubs.clients.http_post_json = aws.build_clients(CREDS, REGION).http_post_json
            provider = AwsProvider(
                CREDS,
                region=REGION,
                lease_ref="lop-ttl-x",
                clients=stubs.clients,
                sleep=stubs.sleep,
            )
            env = _FakeEnv()
            env.pkgs_prefix = "{command}"
            provider._env = env
            provider._public_ip = "127.0.0.1"
            write = f"from pathlib import Path; Path({str(marker)!r}).open('a').write('once\\n')"
            await provider.execute([write])
            with pytest.raises(GuestExecutionError, match=r"exit 7"):
                await provider.execute([write + "; raise SystemExit(7)", write])
            assert marker.read_text() == "once\nonce\n"
            assert len(requests_seen) == 2
            assert stubs.slept == [3.0]
            # Exercise the actual HTTP -> subprocess path, not just a source
            # string assertion: 100k Unicode cannot fit Linux's single-arg cap.
            large_marker = tmp_path / "large-text"
            text = "🙂" * 100_000
            large = f"from pathlib import Path; Path({str(large_marker)!r}).write_text({text!r})"
            await provider.execute([large])
            assert large_marker.read_text() == text
            assert len(requests_seen) == 3
            assert all(len(arg.encode("utf-8")) <= 64_000 for arg in requests_seen[-1]["command"])
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"returncode": 7, "error": "PRIVATE_PAYLOAD"},
        {"error": "PRIVATE_PAYLOAD"},
        {"returncode": False},
        {"returncode": "0"},
        RuntimeError("PRIVATE_PAYLOAD"),
        TimeoutError("PRIVATE_PAYLOAD"),
    ],
)
async def test_execute_failure_stops_batch_without_retry_or_payload_leak(
    response: dict[str, Any] | Exception,
) -> None:
    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        env = _FakeEnv()
        provider._env = env
        provider._public_ip = "127.0.0.1"
        stubs.guest_default = response
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["first()", "must_not_run()"])
        assert "PRIVATE_PAYLOAD" not in str(raised.value)
        assert "batch not retried" in str(raised.value)
        assert len(stubs.guest_posts) == 1
        assert stubs.slept == []
        assert "executed" not in env.kwargs


@pytest.mark.asyncio
async def test_execute_settles_after_the_batch_and_respond_without_simulator_is_none() -> None:
    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        env = _FakeEnv()
        provider._env = env
        provider._public_ip = "127.0.0.1"
        await provider.execute(["pyautogui.click(1, 2)"])
        await provider.execute([])
        # The statement crosses as base64 (see ``python_source_argv``), so assert
        # the endpoint contract and decode the payload rather than pinning a
        # literal argv -- the literal shape is the argv exposure this encoding
        # exists to remove.
        assert len(stubs.guest_posts) == 1
        post = stubs.guest_posts[0]
        assert post["shell"] is False
        command = post["command"]
        assert command[:3] == ["python", "-c", computer_input._SOURCE_BOOTSTRAP]
        assert base64.b64decode("".join(command[3:])).decode("utf-8") == (
            "import pyautogui; pyautogui.click(1, 2)"
        )
        assert "executed" not in env.kwargs
        assert stubs.slept == [3.0, 3.0]
        assert await provider.respond("?") is None
        assert (await provider.observe())["screenshot"] == b"png"


class _OpaqueSimulatorValue:
    def __str__(self) -> str:
        raise AssertionError("private simulator objects must not be stringified")


_SIMULATOR_VALUES = [
    None,
    123,
    {"internal": "PRIVATE_SIMULATOR_CANARY"},
    _OpaqueSimulatorValue(),
    "public answer",
    "",
    "   ",
]
_SIMULATOR_IDS = [
    "refusal",
    "number",
    "mapping",
    "opaque-object",
    "public-string",
    "empty",
    "whitespace",
]


def _answer_adapter(tmp_path: Path, stubs: _Stubs, value: Any) -> Any:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter

    provider = AwsProvider(CREDS, region=REGION, lease_ref="lop-ttl-answer", clients=stubs.clients)
    respond = Mock(return_value=value)
    provider._env = SimpleNamespace(user_simulator=SimpleNamespace(respond=respond))
    adapter = OSWorldV2Adapter(workspace_root=tmp_path)
    # Only the upstream simulator is scripted. No task module, profile, cloud
    # allocation or FakeProvider participates in this answer-bearing boundary.
    adapter._task = taskfile.TaskDescriptor(
        task_id="synthetic",
        instruction="",
        source_sha256="0" * 64,
        user_simulator={"type": "scripted"},
    )
    adapter._provider = provider
    return adapter, respond


@pytest.mark.asyncio
@pytest.mark.parametrize("value", _SIMULATOR_VALUES, ids=_SIMULATOR_IDS)
@pytest.mark.parametrize("boundary", ["provider", "adapter"])
async def test_real_aws_simulator_answer_contract(
    tmp_path: Path, value: Any, boundary: str
) -> None:
    from pydantic import ValidationError

    from local_operator.evaluation.adapters.api import AskUserExchangeParams

    with _Stubs() as stubs:
        adapter, respond = _answer_adapter(tmp_path, stubs, value)
        params = AskUserExchangeParams(
            operation_id="begin", episode_id="episode", ask_id="ask", prompt="Public question?"
        )

        async def call() -> Any:
            if boundary == "provider":
                return await adapter._provider.respond(params.prompt)
            return await adapter.ask_user_exchange(params)

        if value is not None and not isinstance(value, str):
            with pytest.raises(TypeError) as error:
                await call()
            assert str(error.value) == "simulator response must be a string or None"
        elif boundary == "adapter" and isinstance(value, str) and not value.strip():
            with pytest.raises(ValidationError):
                await call()
        else:
            result = await call()
            if boundary == "provider":
                assert result is value
            else:
                assert result.accepted == (value is not None)
                assert result.answer == value and result.request_digest == params.request_digest
        respond.assert_called_once_with("Public question?")
        assert not stubs.http_calls and not stubs.guest_posts


@pytest.mark.asyncio
@pytest.mark.parametrize("value", _SIMULATOR_VALUES, ids=_SIMULATOR_IDS)
async def test_real_aws_answer_never_publishes_coerced_values(
    tmp_path: Path, episode_id: str, value: Any
) -> None:
    from types import SimpleNamespace

    from local_operator.evaluation.evidence.models import UserSimulatorExchangePayload
    from local_operator.evaluation.evidence.verify import verify_bundle
    from local_operator.evaluation.runner.episode import EpisodeRunner
    from tests.unit.evaluation.runner.conftest import (
        FakeAdapter,
        ScriptedModel,
        build_config,
        build_spec,
        payloads,
        selector,
    )

    with _Stubs() as stubs:
        adapter, respond = _answer_adapter(tmp_path, stubs, value)

        class AnswerTransport(FakeAdapter):
            # Reuse the existing no-cloud lifecycle fixture for everything
            # except the real OSWorld -> AwsProvider -> upstream answer path.
            async def handshake(self, *, timeout: float = 10.0) -> Any:
                result = await super().handshake(timeout=timeout)
                metadata = result.metadata.model_copy(
                    update={"capabilities": adapter.metadata.capabilities}
                )
                return result.model_copy(update={"metadata": metadata})

            async def _call_raw(
                self, method: Any, params: Any, result_type: Any, *, timeout: float
            ) -> Any:
                if method == "ask_user_exchange":
                    self.calls.append(method)
                    return await adapter.ask_user_exchange(params)
                return await super()._call_raw(method, params, result_type, timeout=timeout)

        async def rescue(descriptor: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(complete=True)

        transport = AnswerTransport(tmp_path, episode_id)
        model = ScriptedModel(["ask", "finish"])
        outcome = await EpisodeRunner(
            build_spec(episode_id),
            build_config(tmp_path),
            selector=selector(tmp_path),
            model=model,
            launch=lambda _: transport,
            rescue=rescue,
            synthetic_model=True,
        ).run()
        respond.assert_called_once_with("What next?")
        assert transport.calls.count("ask_user_exchange") == 1
        assert outcome.bundle_root is not None
        report = verify_bundle(outcome.bundle_root)
        assert report.valid, report.issues
        exchanges = payloads(outcome.bundle_root, UserSimulatorExchangePayload)
        if isinstance(value, str) and value.strip():
            assert outcome.status == "completed", outcome.diagnostic
            assert len(exchanges) == 1 and model.calls == 2
            answer_file = next(
                path
                for path in outcome.bundle_root.rglob(exchanges[0].response_artifact.sha256)
                if path.is_file()
            )
            assert answer_file.read_text() == value
            assert model.histories[1][0].ask_answer == value
        else:
            assert outcome.status == (
                "cancelled" if value is None else "failed"
            ), outcome.diagnostic
            assert outcome.score is not None and outcome.score.status == "unscored"
            assert not exchanges and model.calls == 1
            assert all(turn.ask_answer is None for history in model.histories for turn in history)
            assert "execute" not in transport.calls and "score" not in transport.calls
            assert "PRIVATE_SIMULATOR_CANARY" not in (outcome.diagnostic or "")
            assert not any(
                b"PRIVATE_SIMULATOR_CANARY" in path.read_bytes()
                for path in outcome.bundle_root.rglob("*")
                if path.is_file()
            )
        assert not stubs.http_calls and not stubs.guest_posts


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def test_audit_lists_instances_volumes_and_schedules_or_empty() -> None:
    adapter_filter = [{"Name": "tag:lop:adapter", "Values": ["osworld-v2"]}]
    with _Stubs() as stubs:
        stubs.ec2_stub.add_response(
            "describe_instances",
            {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": INSTANCE,
                                "State": {"Name": "running"},
                                "Tags": [{"Key": "lop:episode", "Value": EPISODE}],
                            }
                        ]
                    }
                ]
            },
            {
                "Filters": adapter_filter
                + [{"Name": "instance-state-name", "Values": list(aws_mod.LIVE_STATES)}]
            },
        )
        stubs.ec2_stub.add_response(
            "describe_volumes",
            {"Volumes": [{"VolumeId": "vol-1", "State": "available", "Tags": []}]},
            {"Filters": adapter_filter},
        )
        stubs.sched_stub.add_response(
            "list_schedules",
            {"Schedules": [{"Name": f"lop-ttl-{EPISODE}", "State": "ENABLED"}]},
            {"NamePrefix": "lop-ttl-"},
        )
        found = AwsProvider.audit(stubs.clients)
        stubs.ec2_stub.assert_no_pending_responses()
        stubs.sched_stub.assert_no_pending_responses()
    assert [(f["kind"], f["id"], f["episode"]) for f in found] == [
        ("instance", INSTANCE, EPISODE),
        ("volume", "vol-1", None),
        ("schedule", f"lop-ttl-{EPISODE}", EPISODE),
    ]

    with _Stubs() as stubs:
        stubs.ec2_stub.add_response("describe_instances", {"Reservations": []}, {"Filters": ANY})
        stubs.ec2_stub.add_response("describe_volumes", {"Volumes": []}, {"Filters": ANY})
        stubs.sched_stub.add_response(
            "list_schedules", {"Schedules": []}, {"NamePrefix": "lop-ttl-"}
        )
        assert AwsProvider.audit(stubs.clients) == []


# ---------------------------------------------------------------------------
# clients / ttl
# ---------------------------------------------------------------------------


def test_build_clients_pins_the_region_and_the_credential_values() -> None:
    clients = aws_mod.build_clients(CREDS, "us-east-1")
    assert clients.ec2.meta.region_name == "us-east-1"
    assert clients.scheduler.meta.region_name == "us-east-1"
    frozen = clients.ec2._request_signer._credentials
    assert frozen.access_key == "AKIATEST"
    assert frozen.secret_key == "marker-secret"


def test_ttl_seconds_derivation() -> None:
    assert ttl_seconds_for(None) == aws_mod.DEFAULT_TTL_SECONDS
    assert ttl_seconds_for(1_800_000) == 1800 + aws_mod.TTL_SLACK_SECONDS
    assert ttl_seconds_for(0) == aws_mod.TTL_SLACK_SECONDS
    assert ttl_seconds_for(1_800_000, override=60) == aws_mod.TTL_SLACK_SECONDS
    assert ttl_seconds_for(None, override=5000) == 5000


def test_credentials_repr_never_echoes_the_secret() -> None:
    assert "marker-secret" not in repr(CREDS)
    assert "marker-secret" not in str(CREDS)


def _volume_infra(value: str) -> tuple[ScopedInfraValue, ...]:
    return _infra() + (
        ScopedInfraValue(name="AWS_ROOT_VOLUME_SIZE", purpose="benchmark_compute", value=value),
    )


def _describe_images_response(size_gb: int) -> dict[str, Any]:
    return {
        "Images": [
            {
                "ImageId": _plan().ami_id,
                "RootDeviceName": "/dev/sda1",
                "BlockDeviceMappings": [
                    {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": size_gb}},
                ],
            }
        ]
    }


@pytest.mark.asyncio
async def test_a_resolved_root_volume_override_reaches_run_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The override must reach the actual API call, not merely the plan.

    A unit test of the pure function would still pass if ``volume_gb`` were
    dropped on the way to ``run_instances`` -- which is exactly the failure
    that would leave the guest on the 2.2 GB-free disk its recorder exhausts
    at ~t+383s, while every test stayed green. The stubbed client asserts the
    exact launch parameters, so ``VolumeSize`` here is the number AWS would
    actually receive.
    """

    task = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/task_plain.py")
    plan = provisioning.resolve(task, episode_id=EPISODE, infra_values=_volume_infra("120"))
    assert plan.volume_gb == 120

    with _Stubs() as stubs:
        # The floor check reads the AMI, so the lookup still happens -- but now
        # to validate rather than to supply the size.
        _expect_describe_images(stubs)
        _expect_run_instances(stubs, plan=plan, volume_gb=120)
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=_FakeEnv,
            task_factory=lambda t: {"id": t.task_id},
        )
        await provider.allocate(plan, task, cache_root=_cache_root(tmp_path))
        stubs.ec2_stub.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_a_root_volume_smaller_than_the_ami_snapshot_is_refused_before_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EBS cannot restore a snapshot into a smaller volume, so this is doomed.

    AWS refuses it with an InvalidBlockDeviceMapping that names neither size.
    Failing here instead produces a message carrying both numbers and the name
    of the knob to change -- and does it before run_instances, so no instance
    and no volume are ever created.
    """

    task = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/task_plain.py")
    plan = provisioning.resolve(task, episode_id=EPISODE, infra_values=_volume_infra("20"))
    assert plan.volume_gb == 20

    with _Stubs() as stubs:
        # The release AMI's own root snapshot is 30 GiB; 20 cannot hold it.
        stubs.ec2_stub.add_response(
            "describe_images", _describe_images_response(30), {"ImageIds": [_plan().ami_id]}
        )
        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=_FakeEnv,
            task_factory=lambda t: {"id": t.task_id},
        )
        with pytest.raises(AllocationError) as excinfo:
            await provider.allocate(plan, task, cache_root=_cache_root(tmp_path))
        message = str(excinfo.value)
        # Both numbers and the remedy, which is the entire point of the check.
        assert "20" in message and "30" in message
        assert "AWS_ROOT_VOLUME_SIZE" in message
        # No run_instances was queued, and none was attempted.
        stubs.ec2_stub.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_a_root_volume_equal_to_the_ami_snapshot_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The floor is >=, not >. Rejecting an exact match would refuse a size AWS
    accepts, and would be invisible until an operator picked exactly 30."""

    task = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/task_plain.py")
    plan = provisioning.resolve(task, episode_id=EPISODE, infra_values=_volume_infra("30"))

    with _Stubs() as stubs:
        stubs.ec2_stub.add_response(
            "describe_images", _describe_images_response(30), {"ImageIds": [_plan().ami_id]}
        )
        _expect_run_instances(stubs, plan=plan, volume_gb=30)
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=_FakeEnv,
            task_factory=lambda t: {"id": t.task_id},
        )
        await provider.allocate(plan, task, cache_root=_cache_root(tmp_path))
        stubs.ec2_stub.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_the_default_path_still_applies_osworlds_forty_gib_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Splitting the AMI lookup must not change the no-override behaviour.

    ``_resolve_root_volume_size`` keeps OSWorld's 40 GiB floor while the new
    floor CHECK needs the AMI's real 30; a single function serving both would
    have to pick one and be wrong for the other. This pins the default half.
    """

    task = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/task_plain.py")
    plan = _plan()
    assert plan.volume_gb is None

    with _Stubs() as stubs:
        _expect_describe_images(stubs)  # AMI declares 30
        _expect_run_instances(stubs, plan=plan, volume_gb=40)  # floor wins
        _expect_create_schedule(stubs)
        _expect_running(stubs)
        provider = _provider(
            stubs,
            monkeypatch,
            desktop_env_factory=_FakeEnv,
            task_factory=lambda t: {"id": t.task_id},
        )
        await provider.allocate(plan, task, cache_root=_cache_root(tmp_path))
        stubs.ec2_stub.assert_no_pending_responses()

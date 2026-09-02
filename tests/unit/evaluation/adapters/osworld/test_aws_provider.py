"""AwsProvider against botocore's Stubber: the boto3 call shapes, in order.

Nothing here touches the network. Every EC2 / Scheduler call is answered by a
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
    ReadinessTimeout,
    _Clients,
    ttl_seconds_for,
)

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
    """

    for name in ("AWS_PROFILE", "AWS_DEFAULT_REGION", "AWS_REGION"):
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
    """Two stubbed clients plus a scripted HTTP prober and a virtual clock."""

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

        self.clients = _Clients(ec2=self.ec2, scheduler=self.scheduler, http_get=http_get)

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


def _expect_run_instances(stubs: _Stubs) -> None:
    plan = _plan()
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
                        # 40 = OSWorld's floor; the AMI's 30 does not lower it.
                        "VolumeSize": 40,
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
    to cwd would silently re-open the digest-break defect."""

    with _Stubs() as stubs:
        provider = _provider(stubs, monkeypatch, desktop_env_factory=_FakeEnv)
        with pytest.raises(AllocationError, match="cache root"):
            await provider.allocate(_plan(), _task(), cache_root=None)


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
async def test_execute_settles_after_the_batch_and_respond_without_simulator_is_none() -> None:
    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        env = _FakeEnv()
        provider._env = env
        await provider.execute(["pyautogui.click(1, 2)"])
        await provider.execute([])
        assert env.kwargs["executed"] == ["pyautogui.click(1, 2)"]
        assert stubs.slept == [3.0, 3.0]
        assert await provider.respond("?") is None
        assert (await provider.observe())["screenshot"] == b"png"


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

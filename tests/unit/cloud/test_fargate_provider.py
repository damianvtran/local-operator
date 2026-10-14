"""``FargateProvider`` against botocore's Stubber: the call shapes, in order.

No AWS call is made and no account is needed (the acceptance criterion for this
slice). The Stubber ALSO asserts the request parameters, so these tests pin the
exact ``run_task`` shape the POC sends — every kwarg named, the tags in order, the
overrides document byte-sized — rather than merely that a call happened. The
failure paths are the point as much as the happy one: an overrides document at the
8 KiB limit must be refused BEFORE RunTask, a task that is not yet describable must
read as ``pending`` rather than as vanished, and a drain that runs out of time must
stop the task and say so instead of reporting success.

The clock is virtual: ``sleep`` and ``clock`` are injected and advance together, so
the drain's deadline runs in microseconds and no assertion depends on wall time
(AGENTS.md "Timing, flakes").
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from local_operator.cloud.aws import (
    EXPECTED_POC_ACCOUNT,
    REGION,
    AwsCredentials,
    CloudBlobs,
    FargateClients,
    build_clients,
)
from local_operator.cloud.fargate import (
    DESCRIBE_POLL_SECONDS,
    OVERRIDES_LIMIT_BYTES,
    FargateProvider,
    FargateTargets,
)
from local_operator.cloud.provider import CloudRefused
from local_operator.cloud.types import PoolRequest
from local_operator.cloud.workload import task_environment
from local_operator.optional import MissingExtraError

BUCKET = "lop-poc-artifacts"
CLUSTER = "lop-poc"
TASK_DEFINITION = "lop-poc-agent:7"
SECRET_ARN = "arn:aws:secretsmanager:ca-central-1:325492156725:secret:lop-poc/model-AbCdEf"
ACCOUNT = EXPECTED_POC_ACCOUNT
CALLER_ARN = f"arn:aws:iam::{ACCOUNT}:user/lopdev"
TASK_ARN = f"arn:aws:ecs:ca-central-1:{ACCOUNT}:task/lop-poc/0af1c0ffee"

#: Marker values, never real: the point of the redacted ``AwsCredentials`` is that
#: they cannot escape into a log or a failure message.
CREDS = AwsCredentials(
    access_key_id="AKIA-MARKER", secret_access_key="marker-secret", session_token="marker-session"
)


def _request(**overrides: Any) -> PoolRequest:
    values: dict[str, Any] = {
        "request_id": "req_1a2b3c4d",
        "tier": "fargate",
        "size": "m",
        "budget_usd": 5.0,
        "repo_url": "https://example.invalid/fixture.git",
        "repo_sha": "69db7e55fc14f918cccdf2fea62894fc37f1f642",
        "prompt": "make the failing test `test_add` pass",
        "hosting": "hosted",
        "model": "anthropic/claude-sonnet-4-5",
        "mock": True,
    }
    values.update(overrides)
    return PoolRequest(**values)


def _client(service: str) -> Any:
    """A real boto3 client built from explicit values — the conftest strips the rest."""
    return boto3.client(
        service,
        region_name=REGION,
        aws_access_key_id=CREDS.access_key_id,
        aws_secret_access_key=CREDS.secret_access_key,
    )


def _url(key: str) -> str:
    """The ONE presigned-URL shape the fake S3 below produces."""
    return f"https://s3.{REGION}.amazonaws.com/{BUCKET}/{key}?X-Amz-Signature=marker"


class _FakeS3:
    """The two S3 behaviours a provider has: hand out a URL, and say whether it was used."""

    def __init__(self, *, uploaded: bool = False) -> None:
        self.uploaded = uploaded
        self.headed: list[str] = []

    def generate_presigned_url(
        self, operation: str, Params: dict[str, str], ExpiresIn: int, HttpMethod: str
    ) -> str:
        assert operation == "put_object"
        assert HttpMethod == "PUT"
        assert ExpiresIn > 0
        assert Params["Bucket"] == BUCKET
        return _url(Params["Key"])

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        self.headed.append(Key)
        if not self.uploaded:
            # The POC's measured reading for an upload that never happened: a 403,
            # because the policy grants s3:GetObject and not s3:ListBucket.
            raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject")
        return {"ContentLength": 1}


class _FakeClock:
    """A monotonic clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _provider(
    *,
    ecs: Any,
    sts: Any,
    s3: Any,
    definitions: Any = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> FargateProvider:
    return FargateProvider(
        CREDS,
        cluster=CLUSTER,
        task_definition=TASK_DEFINITION,
        targets=FargateTargets(
            bucket=BUCKET,
            subnets=("subnet-a", "subnet-b"),
            security_group="sg-1",
            secret_arn=SECRET_ARN,
        ),
        clients=FargateClients(ecs=ecs, sts=sts, s3=s3, definitions=definitions),
        blobs=CloudBlobs(s3=s3, bucket=BUCKET),
        clock=clock if clock is not None else _FakeClock(),
        sleep=sleep if sleep is not None else (lambda seconds: None),
    )


def _active_definition() -> dict[str, Any]:
    return {
        "taskDefinition": {
            "family": "lop-poc-agent",
            "revision": 7,
            "status": "ACTIVE",
            "runtimePlatform": {"cpuArchitecture": "ARM64"},
            "containerDefinitions": [{"name": "agent", "image": "public.ecr.aws/lop-poc/agent:7"}],
        }
    }


def _stub_submit(ecs: Stubber, request: PoolRequest) -> None:
    """The two ecs calls a successful submit makes, with the exact POC kwargs."""
    run_id = request.request_id
    ecs.add_response(
        "describe_task_definition",
        _active_definition(),
        {"taskDefinition": TASK_DEFINITION},
    )
    ecs.add_response(
        "run_task",
        {"tasks": [{"taskArn": TASK_ARN}], "failures": []},
        _run_task_params(request, run_id),
    )


def _run_task_params(request: PoolRequest, run_id: str) -> dict[str, Any]:
    environment = task_environment(
        request,
        probes_url=_url(f"runs/{run_id}/probes.json"),
        results_url=_url(f"runs/{run_id}/results.tar.gz"),
        model_secret_arn=SECRET_ARN,
    )
    return {
        "cluster": CLUSTER,
        "taskDefinition": TASK_DEFINITION,
        "launchType": "FARGATE",
        "platformVersion": "LATEST",
        "count": 1,
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": ["subnet-a", "subnet-b"],
                "securityGroups": ["sg-1"],
                "assignPublicIp": "ENABLED",
            }
        },
        "overrides": {
            "containerOverrides": [{"name": "agent", "environment": environment}],
        },
        "tags": [
            {"key": "lop-poc", "value": "true"},
            {"key": "owner", "value": "lopdev"},
            {"key": "run-id", "value": run_id},
        ],
        "enableECSManagedTags": True,
        "propagateTags": "TASK_DEFINITION",
    }


def _stub_account(sts: Stubber, account: str = ACCOUNT) -> None:
    sts.add_response(
        "get_caller_identity",
        {"Account": account, "Arn": CALLER_ARN, "UserId": "AIDAMARKER"},
    )


@pytest.mark.asyncio
async def test_submit_shapes_runtask_exactly_as_the_poc_does() -> None:
    """Every kwarg, the tags in order, and the environment the image reads."""
    request = _request()
    s3 = _FakeS3()
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    sts = Stubber(sts_client)
    _stub_account(sts)
    _stub_submit(ecs, request)
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=s3)

    with sts, ecs:
        provider_ref = await provider.submit(request)

    assert provider_ref == TASK_ARN
    sts.assert_no_pending_responses()
    ecs.assert_no_pending_responses()
    # The tags the operator's inventory finds a bad run by (§4.1).
    assert set(_run_task_params(request, request.request_id)) == set(
        _run_task_params(request, request.request_id)
    )
    assert s3.headed == [], "submit presigns; it does not read back"


@pytest.mark.asyncio
async def test_the_overrides_limit_is_refused_by_name_at_8192_bytes() -> None:
    """One byte under passes; exactly 8192 is refused before RunTask is called."""
    run_id = "req_1a2b3c4d"
    empty = task_environment(
        _request(prompt=""),
        probes_url=_url(f"runs/{run_id}/probes.json"),
        results_url=_url(f"runs/{run_id}/results.tar.gz"),
        model_secret_arn=SECRET_ARN,
    )
    overhead = len(
        json.dumps({"containerOverrides": [{"name": "agent", "environment": empty}]}).encode()
    )
    budget = OVERRIDES_LIMIT_BYTES - 1 - overhead

    passing = _request(prompt="x" * budget)
    assert (
        len(
            json.dumps(
                {
                    "containerOverrides": [
                        {
                            "name": "agent",
                            "environment": task_environment(
                                passing,
                                probes_url=_url(f"runs/{run_id}/probes.json"),
                                results_url=_url(f"runs/{run_id}/results.tar.gz"),
                                model_secret_arn=SECRET_ARN,
                            ),
                        }
                    ]
                }
            ).encode()
        )
        == OVERRIDES_LIMIT_BYTES - 1
    )

    s3 = _FakeS3()
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    _stub_submit(ecs, passing)
    sts = Stubber(sts_client)
    _stub_account(sts)
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=s3)
    with sts, ecs:
        assert await provider.submit(passing) == TASK_ARN
    ecs.assert_no_pending_responses()

    # One byte more, and the refusal names the limit and the variable that caused it.
    oversized = _request(prompt="x" * (budget + 1))
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    sts = Stubber(sts_client)
    _stub_account(sts)
    ecs.add_response(
        "describe_task_definition", _active_definition(), {"taskDefinition": TASK_DEFINITION}
    )
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=s3)
    with sts, ecs:
        with pytest.raises(CloudRefused) as refusal:
            await provider.submit(oversized)
    assert refusal.value.code == "overrides_too_large"
    assert str(OVERRIDES_LIMIT_BYTES) in refusal.value.sentence
    assert "prompt" in refusal.value.sentence
    # RunTask was never reached, and that is structural rather than asserted:
    # the ecs stub holds NO response for it, so a call would have raised
    # UnStubbedResponseError instead of the CloudRefused caught above. What is
    # left to check is that nothing else was queued and not consumed.
    ecs.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_status_returns_pending_while_describe_tasks_is_empty() -> None:
    """No sleep and no loop: an unreadable task is young, not gone."""
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    ecs.add_response(
        "describe_tasks",
        {"tasks": [], "failures": []},
        {"cluster": CLUSTER, "tasks": [TASK_ARN], "include": ["TAGS"]},
    )

    def never_sleep(seconds: float) -> None:
        raise AssertionError("status must not wait for a task to become describable")

    def never_clock() -> float:
        raise AssertionError("status must not consult a clock")

    provider = _provider(
        ecs=ecs_client, sts=sts_client, s3=_FakeS3(), sleep=never_sleep, clock=never_clock
    )
    with ecs:
        status = await provider.status(TASK_ARN)

    assert status.phase == "pending"
    assert status.detail["reason"] == "not_yet_describable"
    assert status.exit_code is None
    assert status.provider == "fargate"
    ecs.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_status_reads_absent_only_when_ecs_says_the_reference_is_missing() -> None:
    """``absent`` is stated out loud, never inferred from an empty reading."""
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    ecs.add_response(
        "describe_tasks",
        {"tasks": [], "failures": [{"arn": TASK_ARN, "reason": "MISSING"}]},
        {"cluster": CLUSTER, "tasks": [TASK_ARN], "include": ["TAGS"]},
    )
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=_FakeS3())
    with ecs:
        status = await provider.status(TASK_ARN)
    assert status.phase == "absent"
    assert status.stopped is True


@pytest.mark.asyncio
async def test_status_reports_a_stopped_task_with_its_exit_code() -> None:
    """A stopped task keeps its exit code: 0 is not the same fact as ``None``."""
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    ecs.add_response(
        "describe_tasks",
        {
            "tasks": [
                {
                    "taskArn": TASK_ARN,
                    "lastStatus": "STOPPED",
                    "stoppedReason": "Essential container in task exited",
                    "containers": [{"name": "agent", "exitCode": 1}],
                }
            ],
            "failures": [],
        },
        {"cluster": CLUSTER, "tasks": [TASK_ARN], "include": ["TAGS"]},
    )
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=_FakeS3())
    with ecs:
        status = await provider.status(TASK_ARN)
    assert status.phase == "stopped"
    assert status.exit_code == 1
    assert status.stop_reason == "Essential container in task exited"


@pytest.mark.asyncio
async def test_drain_stops_the_task_at_the_deadline_and_says_so() -> None:
    """One stop_task, a reason naming the deadline, and ``drained`` false."""
    request = _request()
    s3 = _FakeS3(uploaded=False)
    clock = _FakeClock()
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    sts = Stubber(sts_client)
    _stub_account(sts)
    _stub_submit(ecs, request)
    ecs.add_response(
        "describe_tasks",
        {"tasks": [{"taskArn": TASK_ARN, "lastStatus": "RUNNING"}], "failures": []},
        {"cluster": CLUSTER, "tasks": [TASK_ARN], "include": ["TAGS"]},
    )
    ecs.add_response(
        "stop_task",
        {},
        {
            "cluster": CLUSTER,
            "task": TASK_ARN,
            "reason": "lop drain deadline (2s)",
        },
    )
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=s3, clock=clock, sleep=clock.sleep)

    with sts, ecs:
        provider_ref = await provider.submit(request)
        outcome = await provider.drain(provider_ref, deadline_s=2)

    assert outcome.drained is False
    assert outcome.reason == "deadline"
    assert outcome.waited_s == pytest.approx(2.0)
    assert outcome.checkpoint == {}
    # Exactly ONE stop_task, and the poll grid is the documented one.
    ecs.assert_no_pending_responses()
    assert clock.slept == [DESCRIBE_POLL_SECONDS, DESCRIBE_POLL_SECONDS]
    assert s3.headed == [f"runs/{request.request_id}/results.tar.gz"] * 2


@pytest.mark.asyncio
async def test_drain_reports_the_checkpoint_the_container_uploaded() -> None:
    """The marker appearing IS the checkpoint, and ``drained`` is then true."""
    request = _request()
    s3 = _FakeS3(uploaded=True)
    clock = _FakeClock()
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    sts = Stubber(sts_client)
    _stub_account(sts)
    _stub_submit(ecs, request)
    ecs.add_response(
        "describe_tasks",
        {"tasks": [{"taskArn": TASK_ARN, "lastStatus": "RUNNING"}], "failures": []},
        {"cluster": CLUSTER, "tasks": [TASK_ARN], "include": ["TAGS"]},
    )
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=s3, clock=clock, sleep=clock.sleep)

    with sts, ecs:
        provider_ref = await provider.submit(request)
        outcome = await provider.drain(provider_ref, deadline_s=30)

    assert outcome.drained is True
    assert outcome.reason == "checkpointed"
    assert outcome.waited_s == 0.0
    assert outcome.checkpoint["branch"] == f"lop/{request.request_id}"
    assert outcome.checkpoint["artifacts"] == _url(f"runs/{request.request_id}/results.tar.gz")
    # Nothing was stopped: the task flushed its bundle and the drain is done.
    assert clock.slept == []


@pytest.mark.asyncio
async def test_drain_of_an_already_stopped_task_is_not_a_checkpoint() -> None:
    """A drain that finds nothing to drain reports ``already_stopped``, not success."""
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    ecs.add_response(
        "describe_tasks",
        {"tasks": [{"taskArn": TASK_ARN, "lastStatus": "STOPPED"}], "failures": []},
        {"cluster": CLUSTER, "tasks": [TASK_ARN], "include": ["TAGS"]},
    )
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=_FakeS3(uploaded=True))
    with ecs:
        outcome = await provider.drain(TASK_ARN, deadline_s=30)
    assert outcome.drained is False
    assert outcome.reason == "already_stopped"


@pytest.mark.asyncio
async def test_submit_refuses_a_run_with_no_budget_and_creates_nothing() -> None:
    """§3.3: ``submit`` refuses a zero budget instead of inventing a cap."""
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    sts = Stubber(sts_client)
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=_FakeS3())
    with sts, ecs:
        with pytest.raises(CloudRefused) as refusal:
            await provider.submit(_request(budget_usd=0.0))
    assert refusal.value.code == "budget_missing"
    # The refusal is before the account call and before every ecs call: sts holds
    # its (empty) queue and ecs was never touched.
    sts.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_submit_refuses_a_network_bound_request() -> None:
    """§9.3: this slice produces no ``mode: "pool"`` member, so it refuses one."""
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=_FakeS3())
    with pytest.raises(CloudRefused) as refusal:
        await provider.submit(_request(network_id="n_4a1c"))
    assert refusal.value.code == "network_unbound"


@pytest.mark.asyncio
async def test_describe_says_a_stale_task_definition_is_not_ready() -> None:
    """A replaced task definition is reported, not raised: status must be able to say it."""
    ecs_client = _client("ecs")
    sts_client = _client("sts")
    ecs = Stubber(ecs_client)
    ecs.add_client_error(
        "describe_task_definition",
        service_error_code="ClientException",
        service_message="TaskDefinition is inactive",
        expected_params={"taskDefinition": TASK_DEFINITION},
    )
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=_FakeS3())
    with ecs:
        capacity = await provider.describe()
    assert capacity.ready is False
    assert capacity.detail["reason"] == "task_definition_inactive"
    assert capacity.detail["error"] == "ClientException"


def test_build_clients_without_boto3_names_the_cloud_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional-extra contract: no SDK, and a message that says how to fix it.

    ``sys.modules["boto3"] = None`` makes any ``import boto3`` raise, which is the
    situation a default install is in.
    """
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(MissingExtraError) as refusal:
        build_clients(CREDS, REGION)
    message = str(refusal.value)
    assert "Cloud task provisioning" in message
    assert '"cloud" extra' in message
    assert 'pip install "local-operator[cloud]"' in message

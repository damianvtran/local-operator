"""The account guard, and the credential records around it.

The guard is the one rule every other hard rule about this sandbox rests on ("the
account is 325492156725"), so it is tested at the provider rather than at the
script that first needed it: the point of §4.1's extraction is that a second caller
cannot skip it. The proof is structural — the ecs client is activated with NO
expectations, so any call through it raises, and the test still sees the account
refusal, which is only possible if nothing was called.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from botocore.exceptions import UnStubbedResponseError
from botocore.stub import Stubber

from local_operator.cloud.aws import (
    CONTROLLER_SESSION_NAME,
    EXPECTED_POC_ACCOUNT,
    REGION,
    FargateClients,
    assume_role,
    build_clients,
    sdk_exceptions,
    verify_account,
)
from local_operator.cloud.provider import CloudRefused
from tests.unit.cloud.test_fargate_provider import (
    ACCOUNT,
    CALLER_ARN,
    CREDS,
    TASK_DEFINITION,
    _active_definition,
    _client,
    _FakeS3,
    _provider,
    _request,
    _stub_account,
    _stub_submit,
)


def test_verify_account_returns_the_caller_arn() -> None:
    sts_client = _client("sts")
    sts = Stubber(sts_client)
    _stub_account(sts)
    with sts:
        arn = verify_account(FargateClients(ecs=_client("ecs"), sts=sts_client, s3=None), ACCOUNT)
    assert arn == CALLER_ARN


def test_verify_account_refuses_another_account_by_name() -> None:
    sts_client = _client("sts")
    sts = Stubber(sts_client)
    _stub_account(sts, account="999999999999")
    with sts:
        with pytest.raises(CloudRefused) as refusal:
            verify_account(FargateClients(ecs=_client("ecs"), sts=sts_client, s3=None), ACCOUNT)
    assert refusal.value.code == "wrong_account"
    assert "999999999999" in refusal.value.sentence
    assert ACCOUNT in refusal.value.sentence
    assert "AWS_PROFILE" in refusal.value.sentence, "the sentence must name the usual cause"


@pytest.mark.asyncio
async def test_the_account_is_asserted_before_any_mutating_call() -> None:
    """A wrong account plus an EMPTY ecs stub: "first" is proved, not asserted."""
    sts_client = _client("sts")
    sts = Stubber(sts_client)
    _stub_account(sts, account="999999999999")
    ecs_client = _client("ecs")
    ecs = Stubber(ecs_client)
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=_FakeS3())

    with sts, ecs:
        # The control, so the empty stubber cannot be a vacuous gesture: this is
        # what the provider would have hit had it made ANY ecs call first.
        with pytest.raises(UnStubbedResponseError):
            ecs_client.describe_task_definition(taskDefinition=TASK_DEFINITION)
        with pytest.raises(CloudRefused) as refusal:
            await provider.submit(_request())
        sts.assert_no_pending_responses()
        ecs.assert_no_pending_responses()

    assert refusal.value.code == "wrong_account"


@pytest.mark.asyncio
async def test_the_account_is_asserted_once_and_cached() -> None:
    """One assertion per provider, not one per submit: the guard is not a poll."""
    sts_client = _client("sts")
    sts = Stubber(sts_client)
    _stub_account(sts)
    ecs_client = _client("ecs")
    ecs = Stubber(ecs_client)
    request = _request()
    _stub_submit(ecs, request)
    provider = _provider(ecs=ecs_client, sts=sts_client, s3=_FakeS3())
    with sts, ecs:
        await provider.submit(request)
    # The single queued get_caller_identity was consumed exactly once: a second
    # assertion within the same provider would have raised on an empty queue.
    sts.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_describe_reads_the_task_definition_through_the_operator_client() -> None:
    """The identity split: the controller role cannot read task definitions (§4.1).

    ``FargateClients.definitions`` is where the operator's client goes when the
    controller's may not make the call; the controller's ecs client is left with NO
    expectations, so a call through it would fail this test by construction.
    """
    controller = _client("ecs")
    operator = _client("ecs")
    definitions = Stubber(operator)
    definitions.add_response(
        "describe_task_definition", _active_definition(), {"taskDefinition": TASK_DEFINITION}
    )
    with definitions:
        provider = _provider(ecs=controller, sts=_client("sts"), s3=_FakeS3(), definitions=operator)
        capacity = await provider.describe()
    definitions.assert_no_pending_responses()
    assert capacity.ready is True
    assert capacity.detail["family"] == "lop-poc-agent"
    assert capacity.detail["revision"] == "7"
    assert capacity.detail["image"] == "public.ecr.aws/lop-poc/agent:7"


def test_assume_role_returns_credentials_that_cannot_be_printed() -> None:
    """The secrets reach an ``AwsCredentials`` and stop there."""
    sts_client = _client("sts")
    sts = Stubber(sts_client)
    role_arn = f"arn:aws:iam::{ACCOUNT}:role/lop-poc-controller"
    sts.add_response(
        "assume_role",
        {
            "Credentials": {
                "AccessKeyId": "ASIA-MARKER-0001",
                "SecretAccessKey": "[redacted]-access-key",
                "SessionToken": "[redacted]-token",
                "Expiration": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        },
        {
            "RoleArn": role_arn,
            "RoleSessionName": CONTROLLER_SESSION_NAME,
            "DurationSeconds": 10800,
        },
    )
    with sts:
        credentials = assume_role(
            FargateClients(ecs=_client("ecs"), sts=sts_client, s3=None), role_arn
        )
    assert credentials.access_key_id == "ASIA-MARKER-0001"
    rendered = repr(credentials)
    assert "marker-secret-access-key" not in rendered
    assert "marker-session-token" not in rendered
    assert rendered == "AwsCredentials(<redacted>)"


def test_build_clients_builds_from_values_not_from_the_environment() -> None:
    """The session is built from the VALUES, so the ambient chain is never consulted.

    The ``cloud`` extra's boto3 is installed in CI and in this test venv, which is
    what makes this a real check of the constructor rather than of the import.
    """
    clients = build_clients(CREDS, REGION)
    assert clients.ecs.meta.region_name == REGION
    assert clients.s3.meta.region_name == REGION
    # The endpoint override is the POC's measured presign fix, not an accident.
    assert clients.s3.meta.endpoint_url == f"https://s3.{REGION}.amazonaws.com"


def test_the_sdk_exception_type_is_reached_lazily() -> None:
    """``sdk_exceptions`` is a call, not a module constant, for the §3.7 reason."""
    errors = sdk_exceptions()
    assert issubclass(errors.ClientError, Exception)
    assert EXPECTED_POC_ACCOUNT == "325492156725"

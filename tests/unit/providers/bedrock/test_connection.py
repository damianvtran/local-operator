"""T1: descriptors, Region precedence and the env tier (design D6)."""

from __future__ import annotations

import pytest

from local_operator.providers.bedrock import connection as bconn

REF = "BEDROCK_SECRET_" + "a" * 32


@pytest.mark.parametrize(
    "descriptor",
    [
        f"bedrock://api-key?region=us-east-1&ref={REF}",
        "bedrock://profile?name=dev&region=eu-west-1",
        f"bedrock://access-key?region=us-west-2&ref={REF}",
        "bedrock://ambient",
        "bedrock://ambient?region=ap-northeast-1",
        "bedrock://env-bearer",
    ],
)
def test_descriptor_round_trips(descriptor: str) -> None:
    conn = bconn.parse(descriptor)
    assert conn.descriptor() == descriptor
    assert bconn.parse(conn.descriptor()) == conn


def test_raw_string_is_a_bearer_api_key() -> None:
    conn = bconn.parse("bedrock-api-key-abc123")
    assert conn.method == "api-key" and conn.raw_key == "bedrock-api-key-abc123"
    assert conn.uses_bearer
    # The raw key never leaks into the cache scope.
    assert "abc123" not in conn.scope_key()


@pytest.mark.parametrize(
    "bad",
    [
        "bedrock://api-key?region=us-east-1&ref=OPENAI_API_KEY",  # arbitrary operator secret
        "bedrock://api-key?region=us-east-1&ref=BEDROCK_SECRET_xyz",
        "bedrock://access-key?region=us-east-1",
        f"bedrock://profile?name=dev&ref={REF}",
        "bedrock://profile?region=us-east-1",
        "bedrock://ambient?regoin=us-east-1",  # typo must not silently drop the Region
        "bedrock://ambient?region=not a region",
        "bedrock://nope",
    ],
)
def test_malformed_descriptors_are_refused(bad: str) -> None:
    with pytest.raises(bconn.DescriptorError):
        bconn.parse(bad)


def test_stored_region_wins_over_everything() -> None:
    conn = bconn.parse("bedrock://profile?name=dev&region=eu-west-1")
    env = {"AWS_REGION": "us-west-2"}
    values = {"providers": {"bedrock": {"region": "ap-south-1"}}}
    assert bconn.resolve_region(
        conn, profile_region="ca-central-1", values=values, environ=env
    ) == ("eu-west-1")


@pytest.mark.parametrize(
    ("values", "env", "profile", "expected"),
    [
        (
            {"providers": {"bedrock": {"region": "ap-south-1"}}},
            {"AWS_REGION": "us-west-2"},
            "ca-central-1",
            "ap-south-1",
        ),
        (
            {},
            {"AWS_REGION": "us-west-2", "AWS_DEFAULT_REGION": "eu-west-1"},
            "ca-central-1",
            "us-west-2",
        ),
        ({}, {"AWS_DEFAULT_REGION": "eu-west-1"}, "ca-central-1", "eu-west-1"),
        ({}, {}, "ca-central-1", "ca-central-1"),
        ({}, {}, None, "us-east-1"),
    ],
)
def test_region_precedence(values, env, profile, expected) -> None:
    conn = bconn.parse("bedrock://ambient")
    assert (
        bconn.resolve_region(conn, profile_region=profile, values=values, environ=env) == expected
    )


def test_laptop_aws_profile_and_access_keys_do_not_enable_bedrock() -> None:
    """D6: the operator's shell exports these for unrelated production work."""
    env = {
        "AWS_PROFILE": "prod",
        "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "not-a-real-secret",
        "AWS_REGION": "us-east-1",
    }
    assert bconn.bedrock_env_key(env, values={}) is None


@pytest.mark.parametrize(
    ("env", "values", "expected"),
    [
        ({"AWS_BEARER_TOKEN_BEDROCK": "tok"}, {}, "bedrock://env-bearer"),
        ({"LOP_BEDROCK_AUTH": "Ambient"}, {}, "bedrock://ambient"),
        ({"AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://127.0.0.1/creds"}, {}, "bedrock://ambient"),
        ({"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/creds"}, {}, "bedrock://ambient"),
        ({"AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/token"}, {}, "bedrock://ambient"),
        ({"AWS_LAMBDA_FUNCTION_NAME": "fn"}, {}, "bedrock://ambient"),
        ({"ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4"}, {}, "bedrock://ambient"),
        ({}, {"providers": {"bedrock": {"auth": "ambient"}}}, "bedrock://ambient"),
        # Bearer wins over the ambient switch (first match).
        (
            {"AWS_BEARER_TOKEN_BEDROCK": "tok", "LOP_BEDROCK_AUTH": "ambient"},
            {},
            "bedrock://env-bearer",
        ),
        # Any other LOP_BEDROCK_AUTH value is not a switch.
        ({"LOP_BEDROCK_AUTH": "yes"}, {}, None),
    ],
)
def test_env_tier_signals(env, values, expected) -> None:
    assert bconn.bedrock_env_key(env, values=values) == expected


def test_endpoint_override_is_env_only_and_shared_by_both_planes() -> None:
    env = {bconn.ENDPOINT_OVERRIDE_ENV: "http://127.0.0.1:9999/"}
    assert bconn.runtime_base("us-east-1", env) == "http://127.0.0.1:9999"
    assert bconn.control_base("us-east-1", env) == "http://127.0.0.1:9999"
    assert bconn.runtime_base("eu-west-1", {}) == "https://bedrock-runtime.eu-west-1.amazonaws.com"
    assert bconn.control_base("eu-west-1", {}) == "https://bedrock.eu-west-1.amazonaws.com"

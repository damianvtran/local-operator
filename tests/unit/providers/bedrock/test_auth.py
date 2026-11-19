"""T2/T3: credential resolution, error classification and SigV4 parity.

Every case runs against a scratch AWS config (``AWS_CONFIG_FILE`` /
``AWS_SHARED_CREDENTIALS_FILE`` under ``tmp_path``) and with every ambient AWS
variable removed, so nothing reads the operator's ~/.aws, no keychain is
reached and no network call is made (``AWS_EC2_METADATA_DISABLED``).
"""

from __future__ import annotations

import asyncio
import datetime
import time

import pytest

from local_operator.providers.bedrock import auth as bauth
from local_operator.providers.bedrock import connection as bconn
from local_operator.providers.bedrock import errors as berrors

_AMBIENT_VARS = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_BEARER_TOKEN_BEDROCK",
    "LOP_BEDROCK_AUTH",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "LOP_BEDROCK_ENDPOINT_URL",
)


@pytest.fixture(autouse=True)
def scratch_aws(tmp_path, monkeypatch):
    for name in _AMBIENT_VARS:
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "aws-config"
    creds = tmp_path / "aws-credentials"
    config.write_text(
        "[profile dev]\nregion = ca-central-1\n\n"
        "[profile sso]\nsso_start_url = https://example.awsapps.com/start\n"
        "sso_region = us-east-1\nsso_account_id = 111111111111\n"
        "sso_role_name = Dev\nregion = us-west-2\n"
    )
    creds.write_text(
        "[dev]\naws_access_key_id = AKIDDEVPROFILE\naws_secret_access_key = devsecret\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("HOME", str(tmp_path))
    bauth.reset_session_cache()
    yield
    bauth.reset_session_cache()


def test_profile_connection_ignores_ambient_env_credentials(monkeypatch) -> None:
    """An explicit profile must never pick up the shell's production keys."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDAMBIENTPROD")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "prodsecret")
    resolved = bauth.resolve_sync("bedrock://profile?name=dev&region=us-east-1")
    assert resolved.credentials.access_key == "AKIDDEVPROFILE"
    assert resolved.region == "us-east-1"
    assert resolved.bearer is None


def test_profile_region_used_when_descriptor_has_none() -> None:
    conn = bconn.parse("bedrock://profile?name=dev")
    assert bauth.region_for(conn, values={}) == "ca-central-1"


def test_env_bearer_reads_the_env_token(monkeypatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-xyz")
    resolved = bauth.resolve_sync("bedrock://env-bearer", values={})
    assert resolved.bearer == "bedrock-api-key-xyz"
    headers = bauth.sign_headers(resolved, "POST", "https://x/model/m/invoke", b"{}", {})
    assert headers == {"Authorization": "Bearer bedrock-api-key-xyz"}


def test_raw_key_is_a_bearer() -> None:
    resolved = bauth.resolve_sync("raw-bedrock-key", values={})
    assert resolved.bearer == "raw-bedrock-key" and resolved.region == "us-east-1"


def test_sealed_api_key_and_access_key_read_the_store(monkeypatch) -> None:
    ref = "BEDROCK_SECRET_" + "b" * 32
    seen: list[str] = []

    def fake_secret(name: str) -> str:
        seen.append(name)
        return (
            '{"access_key_id":"AKIDSEALED","secret_access_key":"s"}'
            if "access" in seen[0]
            else "tok"
        )

    monkeypatch.setattr(bauth, "_secret_value", lambda r: (seen.append(r), "sealed-token")[1])
    resolved = bauth.resolve_sync(f"bedrock://api-key?region=eu-west-1&ref={ref}", values={})
    assert resolved.bearer == "sealed-token" and seen == [ref]
    monkeypatch.setattr(
        bauth,
        "_secret_value",
        lambda r: '{"access_key_id":"AKIDSEALED","secret_access_key":"s","session_token":"t"}',
    )
    resolved = bauth.resolve_sync(f"bedrock://access-key?region=eu-west-1&ref={ref}", values={})
    assert resolved.credentials.access_key == "AKIDSEALED"
    assert resolved.credentials.token == "t"


def test_missing_sealed_secret_is_a_prewire_auth_error(monkeypatch) -> None:
    monkeypatch.setattr(bauth, "_secret_value", lambda r: None)
    with pytest.raises(bauth.BedrockCredentialError) as caught:
        bauth.resolve_sync("bedrock://api-key?ref=BEDROCK_SECRET_" + "c" * 32, values={})
    error = caught.value.provider_error
    assert error.kind == "auth" and error.request_sent is False and not error.retryable


def test_unknown_profile_maps_to_profile_not_found() -> None:
    with pytest.raises(bauth.BedrockCredentialError) as caught:
        bauth.resolve_sync("bedrock://profile?name=ghost&region=us-east-1")
    error = caught.value.provider_error
    assert error.kind == "auth" and error.request_sent is False
    assert "ghost" in str(error)


def test_sso_profile_with_no_cached_token_fails_before_any_request() -> None:
    """SSO expired/missing: remedy names `aws sso login --profile sso`, nothing sent."""
    with pytest.raises(bauth.BedrockCredentialError) as caught:
        bauth.resolve_sync("bedrock://profile?name=sso&region=us-east-1")
    error = caught.value.provider_error
    assert error.kind == "auth" and error.request_sent is False
    assert "aws sso login --profile sso" in str(error)


def test_ambient_with_nothing_configured_is_no_credentials() -> None:
    with pytest.raises(bauth.BedrockCredentialError) as caught:
        bauth.resolve_sync("bedrock://ambient?region=us-east-1")
    error = caught.value.provider_error
    assert error.kind == "auth" and error.request_sent is False
    assert "AWS_BEARER_TOKEN_BEDROCK" in str(error)


@pytest.mark.parametrize(
    ("class_name", "needle"),
    [
        ("UnauthorizedSSOTokenError", "aws sso login --profile p"),
        ("SSOTokenLoadError", "aws sso login --profile p"),
        ("LoginTokenLoadError", "aws login --profile p"),
        ("ProfileNotFound", "p"),
        ("NoCredentialsError", "p"),
    ],
)
def test_botocore_exceptions_classify_by_name(class_name: str, needle: str) -> None:
    import botocore.exceptions as be

    exc_type = getattr(be, class_name)
    try:
        exc = exc_type(error_msg="x", profile="p")
    except TypeError:
        exc = exc_type()
    error = berrors.classify_credential_error(exc, bconn.parse("bedrock://profile?name=p"))
    assert error.kind == "auth" and error.request_sent is False and not error.retryable
    assert needle in str(error)


def test_slow_credential_resolution_times_out_retryably(monkeypatch) -> None:
    def slow(key, values=None):
        time.sleep(0.5)
        raise AssertionError("should have timed out first")

    monkeypatch.setattr(bauth, "resolve_sync", slow)
    with pytest.raises(bauth.BedrockCredentialError) as caught:
        asyncio.run(bauth.resolve("bedrock://ambient", timeout=0.05))
    error = caught.value.provider_error
    assert error.kind == "timeout" and error.retryable and error.request_sent is False


def test_sigv4_matches_botocore_for_a_colon_model_id(monkeypatch) -> None:
    """T3: canonical URI + Authorization identical to botocore's own client."""
    import botocore.session
    from botocore.credentials import Credentials

    from local_operator.providers.bedrock.client import invoke_url

    fixed = datetime.datetime(2026, 11, 18, 12, 0, 0, tzinfo=datetime.timezone.utc)

    class _Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102
            return fixed if tz else fixed.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):  # noqa: D102
            return fixed.replace(tzinfo=None)

    import botocore.auth

    monkeypatch.setattr(botocore.auth.datetime, "datetime", _Clock)
    model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    body = b'{"anthropic_version":"bedrock-2023-05-31","max_tokens":5,"messages":[]}'
    creds = Credentials("AKIDEXAMPLE", "wJalrXUtnFEMI", "session-token")

    session = botocore.session.Session()
    client = session.create_client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=creds.access_key,
        aws_secret_access_key=creds.secret_key,
        aws_session_token=creds.token,
    )
    captured = {}

    def before_send(request, **_kwargs):
        captured["url"] = request.url
        captured["headers"] = dict(request.headers)
        from botocore.awsrequest import AWSResponse

        return AWSResponse(request.url, 200, {}, None)

    client.meta.events.register("before-send", before_send)
    try:
        client.invoke_model_with_response_stream(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
    except Exception:  # noqa: BLE001 - the fake 200 has no stream; we only need the send
        pass

    url = invoke_url("us-east-1", model_id)
    assert url.endswith(
        "/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/invoke-with-response-stream"
    )
    assert captured["url"] == url

    def text(value):
        return value.decode() if isinstance(value, bytes) else value

    theirs = text(captured["headers"]["Authorization"])
    # botocore adds amz-sdk-*/Content-Length AFTER signing, so sign over exactly
    # the headers its SignedHeaders names — which must be the two our client
    # sends (plus host/x-amz-*), proving the header set matches the SDK too.
    signed = theirs.split("SignedHeaders=", 1)[1].split(",", 1)[0].split(";")
    assert signed == [
        "content-type",
        "host",
        "x-amz-date",
        "x-amz-security-token",
        "x-amzn-bedrock-accept",
    ]
    ours_base = {"Content-Type": "application/json", "X-Amzn-Bedrock-Accept": "application/json"}
    resolved = bauth.ResolvedAuth(region="us-east-1", credentials=creds.get_frozen_credentials())
    ours = bauth.sign_headers(resolved, "POST", url, body, ours_base)
    assert ours["Authorization"] == theirs
    assert ours["X-Amz-Security-Token"] == "session-token"

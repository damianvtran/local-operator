"""Isolated AWS config fixtures for the Bedrock login tests.

The operator's shell exports production AWS variables and has a real
``~/.aws``; NO test may read either. Every test that touches profiles points
``HOME``, ``AWS_CONFIG_FILE``, ``AWS_SHARED_CREDENTIALS_FILE`` and the token
cache directories at ``tmp_path`` and clears every ambient AWS variable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

AMBIENT_VARS = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "LOP_BEDROCK_AUTH",
    "LOP_BEDROCK_ENDPOINT_URL",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_LAMBDA_FUNCTION_NAME",
    "ECS_CONTAINER_METADATA_URI_V4",
    "SSH_CONNECTION",
    "SSH_TTY",
    "AWS_LOGIN_CACHE_DIRECTORY",
)

CONFIG = """\
[default]
region = us-west-2

[profile dev]
region = ca-central-1

[profile corp-sso]
sso_session = corp
sso_account_id = 111122223333
sso_role_name = Dev
region = eu-west-1

[profile legacy-sso]
sso_start_url = https://legacy.awsapps.com/start
sso_region = us-east-1
sso_account_id = 111122223333
sso_role_name = Dev

[profile console]
login_session = arn:aws:iam::111122223333:user/me
region = us-east-2

[profile helper]
credential_process = /bin/false never-run

[profile admin]
role_arn = arn:aws:iam::111122223333:role/Admin
source_profile = dev

[sso-session corp]
sso_start_url = https://corp.awsapps.com/start
sso_region = us-east-1
"""

CREDENTIALS = """\
[default]
aws_access_key_id = AKIDFIXTUREDEFAULT
aws_secret_access_key = fixture-default-secret-never-shown

[dev]
aws_access_key_id = AKIDFIXTUREDEVKEY1
aws_secret_access_key = fixture-dev-secret-never-shown
"""


def iso(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_sso_token(home: Path, key_source: str, *, expires_in: timedelta, refresh: bool) -> None:
    cache = home / ".aws" / "sso" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    token = {
        "accessToken": "fixture-access-token",
        "expiresAt": iso(datetime.now(UTC) + expires_in),
    }
    if refresh:
        token["refreshToken"] = "fixture-refresh"
        token["registrationExpiresAt"] = iso(datetime.now(UTC) + timedelta(days=30))
    name = hashlib.sha1(key_source.encode()).hexdigest()
    (cache / f"{name}.json").write_text(json.dumps(token))


def write_login_token(cache: Path, session: str, *, expires_in: timedelta) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    token = {"accessToken": {"expiresAt": iso(datetime.now(UTC) + expires_in)}}
    (cache / f"{hashlib.sha256(session.encode()).hexdigest()}.json").write_text(json.dumps(token))


def isolate_aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, config: str = CONFIG) -> Path:
    for name in AMBIENT_VARS:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path
    (home / ".aws").mkdir(exist_ok=True)
    (home / ".aws" / "config").write_text(config)
    (home / ".aws" / "credentials").write_text(CREDENTIALS)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(home / ".aws" / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(home / ".aws" / "credentials"))
    monkeypatch.setenv("AWS_LOGIN_CACHE_DIRECTORY", str(home / ".aws" / "login" / "cache"))
    from local_operator.providers.bedrock import auth as bauth
    from local_operator.providers.bedrock import profiles as bprofiles

    bauth.reset_session_cache()
    # NEVER the real AWS CLI from a test: a flow that reaches "Sign in now"
    # would run the operator's ``aws`` and open their browser. Tests that
    # exercise the driver install a fake ``aws`` explicitly after this.
    monkeypatch.setattr(bprofiles, "aws_cli_path", lambda: None)
    return home

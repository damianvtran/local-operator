"""Offline AWS context: profile classification, session state, Regions (design §5.4)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from local_operator.providers.bedrock import profiles as bprofiles
from tests.unit.providers.bedrock.awsfixtures import (
    isolate_aws,
    write_login_token,
    write_sso_token,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return isolate_aws(tmp_path, monkeypatch)


def _by_name(found: list[bprofiles.AwsProfile]) -> dict[str, bprofiles.AwsProfile]:
    return {profile.name: profile for profile in found}


def test_profiles_are_classified_from_config_keys_with_regions(home: Path) -> None:
    found = _by_name(bprofiles.list_profiles())
    assert {name: p.kind for name, p in found.items()} == {
        "default": "static",
        "dev": "static",
        "corp-sso": "sso",
        "legacy-sso": "sso",
        "console": "login",
        "helper": "process",
        "admin": "assume_role",
    }
    assert found["dev"].region == "ca-central-1"
    assert found["admin"].region is None
    assert found["helper"].session.state == "not_applicable"


def test_listing_never_resolves_credentials(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # credential_process can hang or prompt: listing must not reach it.
    import botocore.session

    def boom(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("list_profiles resolved credentials")

    monkeypatch.setattr(botocore.session.Session, "get_credentials", boom)
    assert len(bprofiles.list_profiles()) == 7


def test_context_carries_no_secret_values(home: Path) -> None:
    text = repr([profile.to_json() for profile in bprofiles.list_profiles()])
    assert "never-shown" not in text and "AKIDFIXTURE" not in text


@pytest.mark.parametrize(
    ("expires_in", "refresh", "state"),
    [
        (timedelta(hours=2), True, "valid"),
        (timedelta(hours=-2), True, "expired_refreshable"),
        (timedelta(hours=-2), False, "expired"),
    ],
)
def test_sso_session_state_is_read_from_the_cache(
    home: Path, expires_in: timedelta, refresh: bool, state: str
) -> None:
    write_sso_token(home, "corp", expires_in=expires_in, refresh=refresh)
    session = bprofiles.sso_status("corp-sso")
    assert session.state == state
    assert session.expires_at and session.expires_at.endswith("Z")
    assert session.login_command == "aws sso login --profile corp-sso"


def test_legacy_sso_profile_keys_on_start_url_and_missing_token(home: Path) -> None:
    assert bprofiles.sso_status("legacy-sso").state == "missing"
    write_sso_token(
        home, "https://legacy.awsapps.com/start", expires_in=timedelta(hours=1), refresh=False
    )
    assert bprofiles.sso_status("legacy-sso").state == "valid"


def test_login_session_profile_uses_the_login_cache(home: Path) -> None:
    session = bprofiles.sso_status("console")
    assert session.state == "missing"
    assert session.login_command == "aws login --profile console"
    write_login_token(
        home / ".aws" / "login" / "cache",
        "arn:aws:iam::111122223333:user/me",
        expires_in=timedelta(hours=-1),
    )
    assert bprofiles.sso_status("console").state == "expired"


def test_login_command_quotes_profile_names() -> None:
    assert bprofiles.login_command("team admin", "sso") == "aws sso login --profile 'team admin'"
    assert bprofiles.login_command("dev", "static") is None


@pytest.mark.parametrize(
    ("env", "config", "profile", "expected"),
    [
        ({}, {}, None, ("us-east-1", "fallback")),
        ({}, {}, "eu-west-1", ("eu-west-1", "profile")),
        ({"AWS_DEFAULT_REGION": "us-west-2"}, {}, "eu-west-1", ("us-west-2", "AWS_DEFAULT_REGION")),
        (
            {"AWS_REGION": "ap-south-1", "AWS_DEFAULT_REGION": "us-west-2"},
            {},
            None,
            ("ap-south-1", "AWS_REGION"),
        ),
        (
            {"AWS_REGION": "ap-south-1"},
            {"providers": {"bedrock": {"region": "eu-central-1"}}},
            None,
            ("eu-central-1", "config"),
        ),
    ],
)
def test_default_region_reports_its_source(env, config, profile, expected) -> None:  # noqa: ANN001
    assert (
        bprofiles.default_region(profile_region_value=profile, values=config, environ=env)
        == expected
    )


def test_ambient_sources_name_signals_without_enabling_on_shell_profile() -> None:
    enabled, sources = bprofiles.ambient_sources({"AWS_PROFILE": "prod"}, {})
    assert (enabled, sources) == (False, ["env:AWS_PROFILE"])
    enabled, sources = bprofiles.ambient_sources({"AWS_WEB_IDENTITY_TOKEN_FILE": "/t"}, {})
    assert enabled and sources == ["web_identity"]


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("us-east-1", True),
        ("us-gov-west-1", True),
        ("us-east-123", False),
        ("US-EAST-1", False),
        ("x", False),
    ],
)
def test_region_input_bound_matches_the_desktop_contract(value: str, ok: bool) -> None:
    assert bprofiles.valid_region_input(value) is ok


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("dev", True),
        ("team/admin", True),
        ("me@acme", True),
        ("a]b", False),
        ("x\ny", False),
        ("", False),
        ("a" * 129, False),
    ],
)
def test_profile_input_bound_matches_the_desktop_contract(value: str, ok: bool) -> None:
    assert bprofiles.valid_profile_input(value) is ok


def test_regions_carry_labels_and_geo_prefix() -> None:
    regions = {region["id"]: region for region in bprofiles.regions()}
    assert regions["us-east-1"]["label"] == "US East (N. Virginia)"
    assert regions["eu-west-1"]["geo_prefix"] == "eu"

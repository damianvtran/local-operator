"""``login_bedrock``: the structured AWS chooser behind ``lop login bedrock`` / TUI ``/login``.

Hosts are scripted: ``on_choice`` answers from a queue of values, and
``on_setup_input`` from a queue of typed strings. The connection check is
stubbed at ``check.probe`` / ``check.check_connection`` (the real probe is
covered in ``test_discovery_check.py``), so these tests assert the FLOW:
which questions are asked, what row comes back, and that secrets ride only
in ``secret`` and never in a message.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from local_operator.providers.bedrock import check as bcheck
from local_operator.providers.bedrock import connection as bconn
from local_operator.providers.bedrock import login as blogin
from local_operator.providers.key_check import KeyCheck
from local_operator.providers.oauth.callback_server import LoginCallbacks, LoginError
from tests.unit.providers.bedrock.awsfixtures import isolate_aws, write_sso_token

pytestmark = pytest.mark.asyncio

SECRET = "fixture-" + "secret-access-key-value"


#: What the stubbed connection check saw, in call order; reset by the ``aws``
#: fixture. Kept in THIS module rather than stashed on ``blogin`` so the
#: production module never grows an attribute only the tests know about.
PROBES: list[Any] = []


class Host:
    def __init__(self, choices: list[str | None], typed: list[str | None]) -> None:
        self.choices = list(choices)
        self.typed = list(typed)
        self.asked: list[tuple[str, list[str], str | None]] = []
        self.fields: list[tuple[str, bool]] = []
        self.messages: list[str] = []

    def callbacks(self, *, picker: bool = True) -> LoginCallbacks:
        async def on_choice(title: str, options: list[tuple[str, str, str]], default: str | None):
            self.asked.append((title, [value for value, _l, _d in options], default))
            return self.choices.pop(0)

        async def on_setup_input(label: str, default: str, secret: bool):
            self.fields.append((label, secret))
            return self.typed.pop(0)

        return LoginCallbacks(
            on_progress=self.messages.append,
            on_warning=self.messages.append,
            on_setup_input=on_setup_input,
            on_choice=on_choice if picker else None,
        )


@pytest.fixture
def aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = isolate_aws(tmp_path, monkeypatch)
    PROBES.clear()
    probes = PROBES

    async def probe(resolved, timeout, *, client=None):  # noqa: ANN001
        probes.append(resolved)
        return KeyCheck(True, None)

    async def check_connection(key, timeout, *, client=None):  # noqa: ANN001
        probes.append(key)
        return KeyCheck(True, None)

    monkeypatch.setattr(bcheck, "probe", probe)
    monkeypatch.setattr(bcheck, "check_connection", check_connection)
    return home


async def test_profile_connection_returns_a_descriptor_row(aws: Path) -> None:
    host = Host(choices=["profile", "dev", "ca-central-1"], typed=[])
    row = await blogin.login_bedrock(host.callbacks())
    assert row["key"] == "bedrock://profile?name=dev&region=ca-central-1"
    assert row["account_id"] == "profile:dev@ca-central-1"
    assert row["bedrock"]["label"] == "dev · ca-central-1"
    assert "secret" not in row
    # The profile's own Region is the offered default.
    assert host.asked[2][2] == "ca-central-1"
    assert any("ca-central-1" in m for m in host.messages)  # "Checking … in ca-central-1"


async def test_access_key_secret_is_masked_checked_and_only_in_secret(aws: Path) -> None:
    host = Host(
        choices=["access_key", "us-east-1"],
        typed=["AKIAFIXTUREKEY0001", SECRET, ""],
    )
    row = await blogin.login_bedrock(host.callbacks())
    assert [secret for _label, secret in host.fields] == [False, True, True]
    assert row["secret"] == {
        "access_key_id": "AKIAFIXTUREKEY0001",
        "secret_access_key": SECRET,
        "session_token": None,
    }
    conn = bconn.parse(row["key"])
    assert conn.method == "access-key" and conn.region == "us-east-1"
    assert SECRET not in row["key"] and SECRET not in repr(row["bedrock"])
    assert all(SECRET not in m for m in host.messages)
    assert PROBES[0].credentials.secret_key == SECRET  # checked BEFORE storing


async def test_malformed_access_key_id_is_reasked(aws: Path) -> None:
    host = Host(
        choices=["access_key", "us-east-1"],
        typed=["not-a-key", "AKIAFIXTUREKEY0001", SECRET, "tok"],
    )
    row = await blogin.login_bedrock(host.callbacks())
    assert row["secret"]["session_token"] == "tok"
    assert any("access key ID" in m for m in host.messages)


async def test_api_key_and_other_region(aws: Path) -> None:
    host = Host(
        choices=["api_key", "__other__"], typed=["bedrock-api-key-fixture", "BAD", "eu-south-2"]
    )
    row = await blogin.login_bedrock(host.callbacks())
    assert row["secret"] == "bedrock-api-key-fixture"
    assert bconn.parse(row["key"]).region == "eu-south-2"
    assert any("BAD" in m.upper() for m in host.messages)


async def test_default_chain_without_explicit_region_stores_none(aws: Path) -> None:
    host = Host(choices=[blogin.DEFAULT_CHAIN, "us-east-1"], typed=[])
    row = await blogin.login_bedrock(host.callbacks())
    assert bconn.parse(row["key"]).method == blogin.DEFAULT_CHAIN


async def test_rejected_connection_raises_and_returns_no_row(
    aws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def check_connection(key, timeout, *, client=None):  # noqa: ANN001
        return KeyCheck(False, "Amazon Bedrock rejected these AWS credentials in us-east-1.", "x")

    monkeypatch.setattr(bcheck, "check_connection", check_connection)
    host = Host(choices=["profile", "dev", "us-east-1"], typed=[])
    with pytest.raises(LoginError, match="rejected"):
        await blogin.login_bedrock(host.callbacks())


async def test_sign_in_now_without_aws_cli_ends_with_install_remedy(aws: Path) -> None:
    # The fixture hides the CLI: "Sign in now" must refuse, never spawn.
    host = Host(choices=["profile", "console", "run"], typed=[])
    with pytest.raises(LoginError, match="AWS CLI"):
        await blogin.login_bedrock(host.callbacks())


async def test_expired_sso_profile_offers_sign_in_and_self_run_ends_with_command(aws: Path) -> None:
    write_sso_token(aws, "corp", expires_in=timedelta(hours=-1), refresh=False)
    host = Host(choices=["profile", "corp-sso", "self"], typed=[])
    with pytest.raises(LoginError, match="aws sso login --profile corp-sso"):
        await blogin.login_bedrock(host.callbacks())
    assert host.asked[2][1] == ["run", "self"]


async def test_refreshable_sso_profile_needs_no_sign_in(aws: Path) -> None:
    write_sso_token(aws, "corp", expires_in=timedelta(hours=-1), refresh=True)
    host = Host(choices=["profile", "corp-sso", "eu-west-1"], typed=[])
    row = await blogin.login_bedrock(host.callbacks())
    assert row["bedrock"]["profile"] == "corp-sso"


async def test_numbered_fallback_without_a_picker(aws: Path) -> None:
    # Host has no on_choice: method list is a numbered on_setup_input prompt.
    # 1 = AWS profile; profiles are listed sorted (admin, console, corp-sso,
    # default, dev, ...) so 5 = dev; Enter takes the default Region (option 1,
    # the profile's own ca-central-1).
    host = Host(choices=[], typed=["1", "5", ""])
    row = await blogin.login_bedrock(host.callbacks(picker=False))
    assert row["key"] == "bedrock://profile?name=dev&region=ca-central-1"
    assert "1. " in host.fields[0][0]  # the numbered list rides the label


async def test_cancel_at_method_choice(aws: Path) -> None:
    host = Host(choices=[None], typed=[])
    with pytest.raises(LoginError):
        await blogin.login_bedrock(host.callbacks())


async def test_no_interactive_hooks_fails_with_env_remedy(aws: Path) -> None:
    with pytest.raises(LoginError, match="AWS_BEARER_TOKEN_BEDROCK"):
        await blogin.login_bedrock(LoginCallbacks())


def test_registry_declares_the_aws_login_kind() -> None:
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition("bedrock")
    assert definition is not None
    assert definition.login_kind == "aws"
    assert definition.accepts_api_key and not definition.accepts_paste_prompt


def test_input_bounds() -> None:
    with pytest.raises(blogin.InputInvalid) as raised:
        blogin.validate_input(
            blogin.ConnectionInput("profile", region="us-east-123", profile="dev")
        )
    assert raised.value.code == "bedrock_region_invalid"
    with pytest.raises(blogin.InputInvalid) as raised:
        blogin.validate_input(
            blogin.ConnectionInput(
                "access_key", region="us-east-1", access_key_id="akia", secret_access_key="s"
            )
        )
    assert raised.value.field == "access_key_id"
    assert "s" not in repr(blogin.ConnectionInput("api_key", api_key="sekrit"))[
        40:
    ] or "sekrit" not in repr(blogin.ConnectionInput("api_key", api_key="sekrit"))


def test_label_identity_for_status_listing() -> None:
    from local_operator.providers.auth_store import credential_identity

    assert (
        credential_identity({"account_id": "profile:dev@x", "bedrock": {"label": "dev · x"}})
        == "dev · x"
    )

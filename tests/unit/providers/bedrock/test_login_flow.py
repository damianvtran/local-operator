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
from local_operator.providers.oauth.callback_server import (
    LoginCallbacks,
    LoginCancelledError,
    LoginDeferredError,
    LoginError,
)
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
    host = Host(choices=["sso", "console", "run"], typed=[])
    with pytest.raises(LoginError, match="AWS CLI"):
        await blogin.login_bedrock(host.callbacks())


async def test_expired_sso_profile_offers_sign_in_and_self_run_ends_with_command(aws: Path) -> None:
    # "I'll run it myself" is a CHOICE, not a failure (#4270 D4/Q7): a
    # LoginDeferredError carrying the command apart from the sentence, raised
    # before any row exists (nothing stored), and never a probe.
    write_sso_token(aws, "corp", expires_in=timedelta(hours=-1), refresh=False)
    host = Host(choices=["sso", "corp-sso", "self"], typed=[])
    with pytest.raises(LoginDeferredError) as caught:
        await blogin.login_bedrock(host.callbacks())
    assert host.asked[2][1] == ["run", "self"]
    assert caught.value.command == "aws sso login --profile corp-sso"
    assert "Nothing was stored" in str(caught.value)
    assert not isinstance(caught.value, LoginCancelledError)
    assert PROBES == []


async def test_escape_on_the_sign_in_question_is_a_cancel_not_a_deferral(aws: Path) -> None:
    write_sso_token(aws, "corp", expires_in=timedelta(hours=-1), refresh=False)
    host = Host(choices=["sso", "corp-sso", None], typed=[])
    with pytest.raises(LoginCancelledError):
        await blogin.login_bedrock(host.callbacks())


async def test_refreshable_sso_profile_needs_no_sign_in(aws: Path) -> None:
    write_sso_token(aws, "corp", expires_in=timedelta(hours=-1), refresh=True)
    host = Host(choices=["sso", "corp-sso", "eu-west-1"], typed=[])
    row = await blogin.login_bedrock(host.callbacks())
    assert row["bedrock"]["profile"] == "corp-sso"


async def test_numbered_fallback_without_a_picker(aws: Path) -> None:
    # Host has no on_choice: method list is a numbered on_setup_input prompt.
    # 1 = AWS profile; its list holds only the non-SSO profiles, sorted
    # (admin, default, dev, helper), so 3 = dev; Enter takes the default
    # Region (option 1, the profile's own ca-central-1).
    host = Host(choices=[], typed=["1", "3", ""])
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


# --- #4270 round 1: canonical vocabulary (D1) and the Region "Other…" path (D2) ---


async def test_method_chooser_uses_the_canonical_order_and_labels(aws: Path) -> None:
    # One list shared with the desktop connect dialog; reordering it here
    # alone is the parity break design D1 flagged.
    captured: list[list[tuple[str, str, str]]] = []

    async def on_choice(title, options, default):  # noqa: ANN001
        captured.append(list(options))
        return None

    with pytest.raises(LoginCancelledError):
        await blogin.login_bedrock(
            LoginCallbacks(on_choice=on_choice, on_setup_input=lambda *a: None)
        )
    values = [value for value, _label, _desc in captured[0]]
    labels = [label for _value, label, _desc in captured[0]]
    assert values == ["profile", "sso", "access_key", "api_key", blogin.DEFAULT_CHAIN]
    assert labels == [
        "AWS profile",
        "AWS SSO",
        "Access key",
        "Bedrock API key",
        "Default credentials",
    ]


async def test_aws_profile_lists_non_sso_kinds_and_aws_sso_lists_the_rest(aws: Path) -> None:
    host = Host(choices=["profile", None], typed=[])
    with pytest.raises(LoginCancelledError):
        await blogin.login_bedrock(host.callbacks())
    plain = host.asked[1][1]
    assert "dev" in plain and "admin" in plain and "helper" in plain
    assert not {"corp-sso", "legacy-sso", "console"} & set(plain)

    host = Host(choices=["sso", None], typed=[])
    with pytest.raises(LoginCancelledError):
        await blogin.login_bedrock(host.callbacks())
    assert sorted(host.asked[1][1]) == ["console", "corp-sso", "legacy-sso"]
    assert host.asked[1][0] == "Which AWS SSO profile?"


async def test_kind_labels_are_the_canonical_short_ones() -> None:
    assert [
        blogin.kind_text(kind)
        for kind in ("static", "sso", "login", "assume_role", "process", "web_identity")
    ] == ["Access keys", "SSO", "AWS login", "Assume role", "Credential process", "Web identity"]


async def test_empty_sso_list_says_so_and_returns_to_the_chooser(
    aws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.providers.bedrock import profiles as bprofiles

    plain_only = [p for p in bprofiles.list_profiles() if not blogin.is_sso_profile(p)]
    monkeypatch.setattr(bprofiles, "list_profiles", lambda: plain_only)
    host = Host(choices=["sso", None], typed=[])
    with pytest.raises(LoginCancelledError):
        await blogin.login_bedrock(host.callbacks())
    assert any("aws configure sso" in m for m in host.messages)
    # Back on the method chooser, not cancelled.
    assert len(host.asked) == 2 and host.asked[1][1][0] == "profile"


async def test_other_region_reasks_and_never_discards_a_typed_secret(aws: Path) -> None:
    # Invalid ids re-ask; an empty entry goes back to the Region picker; the
    # API key typed first survives all of it (#4270 D2).
    host = Host(
        choices=["api_key", "__other__", "__other__"],
        typed=["bedrock-api-key-fixture", "nope", "", "ap-southeast-7"],
    )
    row = await blogin.login_bedrock(host.callbacks())
    assert row["secret"] == "bedrock-api-key-fixture"
    assert bconn.parse(row["key"]).region == "ap-southeast-7"
    assert [title for title, _v, _d in host.asked].count(
        "Which AWS Region should Bedrock use?"
    ) == 2
    assert any("nope" in m for m in host.messages)


async def test_other_region_after_max_invalid_returns_to_the_picker(aws: Path) -> None:
    host = Host(
        choices=["api_key", "__other__", "us-east-1"],
        typed=["bedrock-api-key-fixture", "x1", "x2", "x3"],
    )
    row = await blogin.login_bedrock(host.callbacks())
    assert bconn.parse(row["key"]).region == "us-east-1"
    assert row["secret"] == "bedrock-api-key-fixture"

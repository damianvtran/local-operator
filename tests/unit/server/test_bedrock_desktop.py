"""Amazon Bedrock desktop contract (bedrock design §7 + local-operator-ui#1987 deltas).

Driven through the real ``routes/auth.py`` router on the shared desktop
fixture. Every test isolates AWS config into ``tmp_path`` (``isolate_aws``),
and the signed probe is stubbed at ``check.probe`` / ``check.check_connection``
so no request leaves the process.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from local_operator.providers.bedrock import check as bcheck
from local_operator.providers.key_check import KeyCheck
from tests.unit.providers.bedrock.awsfixtures import isolate_aws, write_sso_token
from tests.unit.server.test_desktop_controls import desktop as _desktop_fixture

desktop = _desktop_fixture
pytestmark = pytest.mark.asyncio

SECRET = "fixture-" + "secret-access-key-PLAINTEXT"


@pytest.fixture
def aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, KeyCheck]:
    isolate_aws(tmp_path, monkeypatch)
    verdict = {"value": KeyCheck(True, None)}

    async def probe(resolved, timeout, *, client=None):  # noqa: ANN001
        return verdict["value"]

    async def check_connection(key, timeout, *, client=None):  # noqa: ANN001
        return verdict["value"]

    monkeypatch.setattr(bcheck, "probe", probe)
    monkeypatch.setattr(bcheck, "check_connection", check_connection)
    return verdict


def _detail(response) -> dict[str, Any]:  # noqa: ANN001
    return response.json()["detail"]


async def test_feature_flag_is_advertised() -> None:
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["bedrock_auth"] == 1


async def test_generic_login_is_refused_with_use_aws_panel(desktop, aws) -> None:  # noqa: ANN001
    client, _app = desktop
    response = await client.post("/v1/auth/login", json={"provider": "bedrock"})
    assert response.status_code == 422
    assert _detail(response)["code"] == "bedrock_use_aws_panel"


async def test_census_advertises_the_aws_method(desktop, aws) -> None:  # noqa: ANN001
    client, _app = desktop
    rows = (await client.get("/v1/auth/providers")).json()["result"]
    row = next(
        r for r in (rows["providers"] if isinstance(rows, dict) else rows) if r["id"] == "bedrock"
    )
    kinds = {m["method_id"]: m["kind"] for m in row["auth_methods"]}
    assert kinds == {"bedrock": "aws", "bedrock:api-key": "api_key"}
    assert row["login_kind"] == "aws"


async def test_context_lists_profiles_regions_and_no_secrets(
    desktop, aws, tmp_path
) -> None:  # noqa: ANN001
    client, _app = desktop
    write_sso_token(tmp_path, "corp", expires_in=timedelta(hours=-1), refresh=False)
    response = await client.get("/v1/auth/providers/bedrock/aws")
    assert response.status_code == 200
    context = response.json()["result"]
    assert set(context) == {
        "regions",
        "default_region",
        "default_region_source",
        "profiles",
        "aws_cli_available",
        "ambient",
        "connections",
    }
    assert context["default_region"] == "us-east-1"
    assert context["default_region_source"] == "fallback"
    corp = next(p for p in context["profiles"] if p["name"] == "corp-sso")
    assert corp["kind"] == "sso" and corp["region"] == "eu-west-1"
    assert corp["session"]["state"] == "expired"
    assert corp["session"]["login_command"] == "aws sso login --profile corp-sso"
    assert context["ambient"] == {"enabled": False, "sources": []}
    assert context["aws_cli_available"] is False  # fixture hides the CLI
    text = json.dumps(context)
    assert "never-shown" not in text and "AKIDFIXTURE" not in text


async def test_broken_config_answers_context_failed_with_parser_text(
    desktop, tmp_path, monkeypatch  # noqa: ANN001
) -> None:
    isolate_aws(tmp_path, monkeypatch, config="[profile dev\nregion = x\n")
    client, _app = desktop
    response = await client.get("/v1/auth/providers/bedrock/aws")
    assert response.status_code == 500
    detail = _detail(response)
    assert detail["code"] == "bedrock_context_failed"
    assert "config" in detail["text"].lower() or "pars" in detail["text"].lower()


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (
            {"method": "profile", "profile": "dev", "region": "us-east-123"},
            "bedrock_region_invalid",
        ),
        ({"method": "profile", "profile": "a]b", "region": "us-east-1"}, "bedrock_input_invalid"),
        (
            {
                "method": "access_key",
                "region": "us-east-1",
                "access_key_id": "akid",
                "secret_access_key": "s",
                "session_token": None,
            },
            "bedrock_input_invalid",
        ),
        (
            {"method": "profile", "profile": "dev", "region": "us-east-1", "api_key": "x"},
            "bedrock_input_invalid",
        ),
        ({"method": "nope", "region": "us-east-1"}, "bedrock_input_invalid"),
    ],
)
async def test_connect_bounds_answer_422_with_codes(
    desktop, aws, body, code
) -> None:  # noqa: ANN001
    client, _app = desktop
    response = await client.put("/v1/auth/providers/bedrock/connection", json=body)
    assert response.status_code == 422
    assert _detail(response)["code"] == code


async def test_connect_access_key_stores_sealed_row_and_lists_it(
    desktop, aws, tmp_path
) -> None:  # noqa: ANN001
    client, _app = desktop
    response = await client.put(
        "/v1/auth/providers/bedrock/connection",
        json={
            "method": "access_key",
            "region": "us-east-1",
            "access_key_id": "AKIDFIXTURE1234567",
            "secret_access_key": SECRET,
            "session_token": None,
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["valid"] is True and result["reason"] is None
    assert result["label"] == "AKID…4567 · us-east-1"
    assert SECRET not in response.text
    # Every auth.db file (main + WAL): the secret must be in none of them.
    db_files = list(tmp_path.glob("auth.db*"))
    assert db_files
    for path in db_files:
        assert SECRET.encode() not in path.read_bytes()
    context = (await client.get("/v1/auth/providers/bedrock/aws")).json()["result"]
    assert context["connections"] == [
        {
            "credential_id": result["credential_id"],
            "method": "access_key",
            "profile": None,
            "region": "us-east-1",
            "label": "AKID…4567 · us-east-1",
        }
    ]


async def test_connect_unverified_has_reason_and_no_code(desktop, aws) -> None:  # noqa: ANN001
    client, _app = desktop
    aws["value"] = KeyCheck(None, "Could not reach Amazon Bedrock in eu-west-1.")
    response = await client.put(
        "/v1/auth/providers/bedrock/connection",
        json={"method": "profile", "profile": "dev", "region": "eu-west-1"},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["valid"] is None and "Could not reach" in result["reason"]
    assert "code" not in result


async def test_connect_rejection_carries_the_bedrock_code_and_stores_nothing(
    desktop, aws, tmp_path
) -> None:  # noqa: ANN001
    client, _app = desktop
    aws["value"] = KeyCheck(
        False, "The AWS sign-in session for profile corp-sso has expired.", "bedrock_sso_expired"
    )
    response = await client.put(
        "/v1/auth/providers/bedrock/connection",
        json={"method": "profile", "profile": "corp-sso", "region": "eu-west-1"},
    )
    assert response.status_code == 422
    assert _detail(response)["code"] == "bedrock_sso_expired"
    context = (await client.get("/v1/auth/providers/bedrock/aws")).json()["result"]
    assert context["connections"] == []


async def test_check_route_reports_code_only_for_rejection(desktop, aws) -> None:  # noqa: ANN001
    client, _app = desktop
    stored = (
        await client.put(
            "/v1/auth/providers/bedrock/connection",
            json={"method": "profile", "profile": "dev", "region": "us-east-1"},
        )
    ).json()["result"]
    aws["value"] = KeyCheck(False, "rejected", "bedrock_credentials_rejected")
    response = await client.post(
        "/v1/auth/providers/bedrock/check", json={"credential_id": stored["credential_id"]}
    )
    assert response.json()["result"] == {
        "valid": False,
        "reason": "rejected",
        "code": "bedrock_credentials_rejected",
    }
    missing = await client.post("/v1/auth/providers/bedrock/check", json={"credential_id": 9999})
    assert missing.status_code == 404
    assert _detail(missing)["code"] == "bedrock_connection_missing"


async def test_sso_login_refusals(desktop, aws) -> None:  # noqa: ANN001
    client, _app = desktop
    response = await client.post(
        "/v1/auth/providers/bedrock/sso-login", json={"profile": "corp-sso"}
    )
    assert response.status_code == 422 and _detail(response)["code"] == "aws_cli_missing"
    response = await client.post("/v1/auth/providers/bedrock/sso-login", json={"profile": "dev"})
    assert _detail(response)["code"] == "bedrock_sso_not_applicable"
    response = await client.post("/v1/auth/providers/bedrock/sso-login", json={"profile": "ghost"})
    assert _detail(response)["code"] == "bedrock_profile_not_found"


def _install_fake_aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> Path:
    import sys

    from local_operator.providers.bedrock import profiles as bprofiles
    from tests.unit.providers.bedrock.test_sso_login import FAKE_AWS

    script = tmp_path / "bin" / "aws"
    script.parent.mkdir(exist_ok=True)
    script.write_text(FAKE_AWS.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_AWS_LOG", str(tmp_path / "aws.log"))
    monkeypatch.setenv("FAKE_AWS_MODE", mode)
    monkeypatch.setattr(bprofiles, "aws_cli_path", lambda: str(script))
    return script


async def _poll(client, op_id: str, states: set[str]) -> dict[str, Any]:  # noqa: ANN001
    import asyncio

    snap: dict[str, Any] = {}
    for _ in range(200):
        snap = (await client.get(f"/v1/auth/operations/{op_id}")).json()["result"]
        if snap["state"] in states:
            return snap
        await asyncio.sleep(0.05)
    raise AssertionError(f"operation never reached {states}: {snap}")


async def test_sso_login_operation_publishes_device_url_and_code_then_succeeds(
    desktop, aws, tmp_path, monkeypatch  # noqa: ANN001
) -> None:
    client, _app = desktop
    _install_fake_aws(tmp_path, monkeypatch, "ok")
    response = await client.post(
        "/v1/auth/providers/bedrock/sso-login", json={"profile": "corp-sso"}
    )
    assert response.status_code == 200, response.text
    first = response.json()["result"]
    assert first["provider"] == "bedrock"
    seen = first
    if first["state"] not in ("succeeded",):
        seen = await _poll(client, first["id"], {"waiting", "succeeded"})
    if seen["state"] == "waiting":
        assert seen["auth_url"] == "https://device.sso.us-east-1.amazonaws.com/"
        assert seen["user_code"] == "WXYZ-ABCD"
    done = await _poll(client, first["id"], {"succeeded"})
    assert "corp-sso" in done["message"]
    assert done["auth_url"] is None and done["user_code"] is None


async def test_sso_login_busy_then_cancel_reaps_the_cli(
    desktop, aws, tmp_path, monkeypatch  # noqa: ANN001
) -> None:
    import asyncio
    import os

    client, _app = desktop
    _install_fake_aws(tmp_path, monkeypatch, "cancel")
    first = (
        await client.post("/v1/auth/providers/bedrock/sso-login", json={"profile": "corp-sso"})
    ).json()["result"]
    waiting = await _poll(client, first["id"], {"waiting"})
    assert waiting["user_code"] == "WXYZ-ABCD"
    busy = await client.post("/v1/auth/providers/bedrock/sso-login", json={"profile": "corp-sso"})
    assert busy.status_code == 409 and _detail(busy)["code"] == "bedrock_sso_busy"
    cancelled = await client.delete(f"/v1/auth/operations/{first['id']}")
    assert cancelled.json()["result"]["state"] == "cancelled"
    child = int((tmp_path / "aws.log.child").read_text())
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("the AWS CLI's process group survived the cancel")

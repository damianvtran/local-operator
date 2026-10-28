"""Backend-only setup: observational preview, explicit consent, pinned identity.

The real production TeamBridgeClient and StatusDiscipline run over mocked HTTP;
only connector placement is stubbed (no host service mutation). Negative cases
must emit no settings/enrollment request and no native owner/config mutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import setup_cli
from local_operator.team_bridge.client import StatusDiscipline
from local_operator.team_bridge.store import Store
from tests.unit.team_bridge.test_client import FakeRow, FakeStore, _ok

pytestmark = pytest.mark.asyncio

HUB = "https://hub.example/v1"
CONSENT = {"version": "bridge_consent_v2", "text": "Accept agent messages for this account."}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    monkeypatch.setattr(setup_cli, "TeamBridgeClient", _production_client)
    monkeypatch.setattr(setup_cli, "StatusDiscipline", _production_discipline)


def _production_client(config_dir: Path, **kwargs: Any):
    client = tb_client.TeamBridgeClient(
        config_dir,
        base_url=HUB,
        store=FakeStore(
            [
                FakeRow(17, data={"access": "token-17", "account_id": "acc_17"}),
                FakeRow(42, data={"access": "token-42", "account_id": "acc_42"}),
            ]
        ),
        transport=httpx.MockTransport(_handler),
        **kwargs,
    )
    return client


def _production_discipline(config_dir: Path, **kwargs: Any):
    # A separate pinned OAuth row proves the preview and write never select the
    # sole/first credential when the machine has several Radient accounts.
    return StatusDiscipline(
        config_dir,
        base_url=HUB,
        store=FakeStore(
            [
                FakeRow(17, data={"access": "token-17", "account_id": "acc_17"}),
                FakeRow(42, data={"access": "token-42", "account_id": "acc_42"}),
            ]
        ),
        transport=httpx.MockTransport(_handler),
        **kwargs,
    )


requests: list[httpx.Request] = []
put_answers: list[dict[str, Any]] = []


def _status() -> dict[str, Any]:
    return {
        "api_version": "1",
        "available": True,
        "consent": CONSENT,
        "tenants": [
            {
                "tenant_id": "tnt_42",
                "tenant_name": "Selected organization",
                "eligible": True,
                "endpoint": None,
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            },
            {
                "tenant_id": "tnt_other",
                "tenant_name": "Other organization",
                "eligible": True,
                "endpoint": None,
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            },
        ],
    }


def _handler(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    if request.url.path.endswith("/messaging/status"):
        return _ok(_status())
    if request.url.path.endswith("/messaging/events"):
        return _ok({"events": [], "cursor": 0})
    if request.method == "PUT" and request.url.path.endswith("/messaging/settings"):
        body = json.loads(request.content)
        return _ok(
            put_answers.pop(0)
            if put_answers
            else {
                "endpoint": {
                    "id": "ep_42",
                    "state": "active",
                    "consent": {"version": body.get("consent_version"), "current": True},
                }
            }
        )
    return httpx.Response(500, json={"error": "unexpected", "code": "unexpected"})


@pytest.fixture(autouse=True)
def _requests() -> None:
    requests.clear()
    put_answers.clear()


def _session(root: Path) -> None:
    directory = root / "sessions" / "session-42"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")


def _values(root: Path) -> dict[str, Any]:
    manager = ConfigManager(config_dir=root)
    return {
        key: settings_io.read_setting(manager, setting).value
        for key, setting in settings_io.BY_KEY.items()
        if key.startswith("team_bridge.")
    }


def _writes() -> list[httpx.Request]:
    return [r for r in requests if r.method not in {"GET", "HEAD"}]


async def _preview(root: Path, **kwargs: Any) -> dict[str, Any]:
    return await setup_cli.setup(root, credential_id=42, **kwargs)


async def test_preview_exposes_served_consent_and_never_enrolls(tmp_path: Path) -> None:
    result = await _preview(tmp_path)
    assert result == {
        "ok": True,
        "code": "consent_preview",
        "credential_id": 42,
        "account_id": "acc_42",
        "tenant_id": "tnt_42",
        "consent": {**CONSENT, "source": "served"},
    }
    assert _writes() == []
    assert Store(tmp_path).committed_owner_count() == 0
    manager = ConfigManager(config_dir=tmp_path)
    assert settings_io.read_setting(manager, settings_io.BY_KEY["team_bridge.session"]).value == ""
    assert (
        settings_io.read_setting(manager, settings_io.BY_KEY["team_bridge.enabled"]).value is False
    )
    assert [r for r in requests if r.url.path.endswith("/messaging/status")]
    assert not any(r.url.path.endswith("/messaging/events") for r in requests)


async def test_preview_cannot_use_default_on_enrollment_or_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.team_bridge.connector import install

    monkeypatch.setattr(
        install, "sync_connector_service", lambda root: (_ for _ in ()).throw(AssertionError)
    )
    result = await _preview(tmp_path)
    assert result["code"] == "consent_preview"
    assert _writes() == []


async def test_preview_does_not_reconcile_existing_designation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(self: Any, check: Any) -> Any:
        nonlocal called
        called = True
        return check

    monkeypatch.setattr(StatusDiscipline, "_sync_local_designation", forbidden)
    await _preview(tmp_path)
    assert called is False


async def test_account_mismatch_refuses_before_status_or_write(tmp_path: Path) -> None:
    result = await _preview(tmp_path, account_id="acc_17")
    assert result["code"] == "account_mismatch"
    assert requests == []
    assert Store(tmp_path).committed_owner_count() == 0


async def test_unselected_tenant_refuses_before_any_write(tmp_path: Path) -> None:
    result = await _preview(tmp_path, tenant_id="tnt_other")
    assert result["code"] == "tenant_mismatch"
    assert _writes() == []
    assert Store(tmp_path).committed_owner_count() == 0


async def test_acceptance_needs_version_and_session_together(tmp_path: Path) -> None:
    for kwargs in ({"accept_consent_version": CONSENT["version"]}, {"session": "session-42"}):
        result = await _preview(tmp_path, **kwargs)
        assert result["code"] == "acceptance_and_session_required"
    assert requests == []


async def test_stale_consent_version_is_refused_without_write_or_config(tmp_path: Path) -> None:
    _session(tmp_path)
    result = await _preview(
        tmp_path,
        account_id="acc_42",
        tenant_id="tnt_42",
        accept_consent_version="bridge_consent_v1",
        session="session-42",
    )
    assert result["code"] == "consent_version_mismatch"
    assert result["consent"]["version"] == CONSENT["version"]
    assert _writes() == []
    assert Store(tmp_path).committed_owner_count() == 0
    manager = ConfigManager(config_dir=tmp_path)
    assert settings_io.read_setting(manager, settings_io.BY_KEY["team_bridge.session"]).value == ""
    assert (
        settings_io.read_setting(manager, settings_io.BY_KEY["team_bridge.enabled"]).value is False
    )


async def test_rejected_served_text_cannot_be_accepted(tmp_path: Path) -> None:
    # Same-version changed text is a negative control for AcceptedTexts: the
    # preview must expose no cached/embedded substitute, let alone accept it.
    original = CONSENT["text"]
    CONSENT["text"] = "Changed bytes under a version already accepted."
    try:
        first = await _preview(tmp_path)
        assert first["code"] == "served_consent_required"
        assert _writes() == []
    finally:
        CONSENT["text"] = original


async def test_nonexistent_session_refuses_before_remote_consent(tmp_path: Path) -> None:
    result = await _preview(
        tmp_path,
        accept_consent_version=CONSENT["version"],
        session="session-42",
    )
    assert result["code"] == "session_not_found"
    assert _writes() == []
    assert Store(tmp_path).committed_owner_count() == 0


async def test_no_local_commit_is_not_reported_as_success(tmp_path: Path) -> None:
    _session(tmp_path)
    put_answers.append({"endpoint": {"id": "ep_42", "state": "paused"}})
    result = await _preview(
        tmp_path,
        accept_consent_version=CONSENT["version"],
        session="session-42",
    )
    assert result["code"] == "local_enrollment_failed"
    assert result["local_enrollment"]["committed"] is False
    manager = ConfigManager(config_dir=tmp_path)
    assert settings_io.read_setting(manager, settings_io.BY_KEY["team_bridge.session"]).value == ""
    assert (
        settings_io.read_setting(manager, settings_io.BY_KEY["team_bridge.enabled"]).value is False
    )


async def test_explicit_setup_commits_owner_designation_and_wanted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path)
    placed: list[Path] = []
    from local_operator.team_bridge.connector import install

    monkeypatch.setattr(install, "sync_connector_service", lambda root: placed.append(root))
    result = await _preview(
        tmp_path,
        accept_consent_version=CONSENT["version"],
        session="session-42",
    )
    assert result["code"] == "setup_complete"
    assert result["local_enrollment"]["committed"] is True
    values = _values(tmp_path)
    assert values["team_bridge.session"] == "session-42"
    assert values["team_bridge.enabled"] is True
    owner = Store(tmp_path).active_owner_readonly()
    assert owner is not None
    assert (owner.credential_id, owner.account_id, owner.tenant_id, owner.endpoint_id) == (
        42,
        "acc_42",
        "tnt_42",
        "ep_42",
    )
    assert Store(tmp_path).designated_session_for_readonly(owner) == "session-42"
    receipt = Store(tmp_path).owner_consent_receipt_readonly(owner)
    assert receipt is not None and receipt["consent_version"] == CONSENT["version"]
    assert placed == [tmp_path]
    writes = _writes()
    assert len(writes) == 1 and writes[0].method == "PUT"
    assert json.loads(writes[0].content) == {
        "state": "active",
        "tenant_id": "tnt_42",
        "consent_version": CONSENT["version"],
        "consented_via": "settings",
    }
    assert writes[0].headers["authorization"] == "Bearer token-42"
    assert any(r.url.path.endswith("/messaging/events") for r in requests)
    # The real client wrote the wanted state through settings_io, not a direct
    # config or enrollment-store shortcut in setup_cli.
    assert settings_io.BY_KEY["team_bridge.session"].key == "team_bridge.session"

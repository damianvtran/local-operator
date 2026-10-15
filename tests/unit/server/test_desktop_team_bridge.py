"""The five agent-messages desktop routes: state shapes, refusals, and the gate.

The bridge CLIENT is faked here on purpose — its own behaviour (auth refresh,
refusal mapping, body rules) is exercised byte-for-byte against a scripted
transport in ``tests/unit/team_bridge``. These tests pin the ROUTE contract the
UI lane consumes: what a success answers, how each refusal class maps to a
status and a ``code``, and that the consent-conflict path invalidates the shared
status cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.providers.auth_store import OAuthAccess
from local_operator.server.routes import desktop_team_bridge
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import (
    STATE_READY,
    STATE_SIGNED_OUT,
    AcceptedText,
    BridgeRefusal,
    BridgeSignInRequired,
    StatusCheck,
)
from local_operator.team_bridge.screening import validate_envelope

# ``_owner_key`` is private, and deliberately reached for: it is the store's OWN
# derivation of the key its rows are written under, and a copy of it here would be
# a second spelling of the thing these fixtures must match.
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    _owner_key,
    state_db_path,
)
from local_operator.team_bridge.types import ScreeningReceipt

pytestmark = pytest.mark.asyncio

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeClient:
    """Scripted per-method behaviour; records every constructed instance."""

    behavior: dict[str, Any] = {}
    instances: list["FakeClient"] = []
    #: The committed hub of the fixtures below. A REAL client always names its
    #: destination, and the read path refuses one that cannot (the shared
    #: destination guard), so a double that omitted this would exercise the
    #: refusal instead of the read.
    base_url = "https://hub.example"

    def __init__(self, config_dir: Any = None, **kwargs: Any) -> None:
        self.config_dir = config_dir
        FakeClient.instances.append(self)

    def _answer(self, name: str) -> Any:
        value = FakeClient.behavior.get(name, {})
        if isinstance(value, Exception):
            raise value
        return value

    async def status(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("status")

    async def current_access(self) -> Any:
        return self._answer("access")

    async def inbox(
        self,
        *,
        tenant_id: str,
        after: str | None = None,
        limit: int | None = None,
        state: str | None = None,
        thread_id: str | None = None,
        sender_account_id: str | None = None,
        access: Any = None,
    ) -> dict[str, Any]:
        FakeClient.behavior["last_inbox_kwargs"] = {
            "tenant_id": tenant_id,
            "after": after,
            "limit": limit,
            "state": state,
            "thread_id": thread_id,
            "sender_account_id": sender_account_id,
            "access": access,
        }
        # A hook that runs INSIDE the wire call (review Q4): the only window in
        # which local state can move after the standing check and before the page
        # is re-validated. Without it, deleting the post-await ``_recheck`` left
        # the whole suite green.
        during = FakeClient.behavior.get("during_inbox")
        if callable(during):
            during()
        return self._answer("inbox")

    async def message(
        self, message_id: str, *, tenant_id: str, access: Any = None
    ) -> dict[str, Any]:
        FakeClient.behavior["last_message_kwargs"] = {
            "message_id": message_id,
            "tenant_id": tenant_id,
            "access": access,
        }
        # The single-message read's own in-flight hook, for the same reason the
        # inbox has one: the window between the standing check and the post-await
        # re-check is where a re-enrollment can land, and a cell that cannot move
        # state inside it cannot prove the re-check is load-bearing.
        during = FakeClient.behavior.get("during_message")
        if callable(during):
            during()
        return self._answer("message")

    async def settings(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        return self._answer("settings")

    async def put_settings(self, **kwargs: Any) -> dict[str, Any]:
        FakeClient.behavior["last_put_kwargs"] = kwargs
        return self._answer("put_settings")

    async def allowlist(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        return self._answer("allowlist")

    async def put_allowlist(
        self,
        entries: Any,
        *,
        tenant_id: str | None = None,
        expected_account_id: str | None = None,
    ) -> dict[str, Any]:
        FakeClient.behavior["last_entries"] = list(entries)
        FakeClient.behavior["last_allowlist_kwargs"] = {
            "tenant_id": tenant_id,
            "expected_account_id": expected_account_id,
        }
        return self._answer("put_allowlist")


class FakeDiscipline:
    def __init__(self, check: StatusCheck) -> None:
        self.check_result = check
        self.boundaries: list[str] = []
        self.invalidated = 0

    async def check(self, boundary: Any) -> StatusCheck:
        self.boundaries.append(str(boundary))
        # Mirror the real discipline: the answer carries the boundary that
        # produced it (a cache hit re-labels the same payload).
        return replace(self.check_result, boundary=str(boundary))

    def invalidate(self) -> None:
        self.invalidated += 1


def _status_check(**overrides: Any) -> StatusCheck:
    base: dict[str, Any] = dict(
        state=STATE_READY,
        boundary="surface_open",
        payload={
            "api_version": "1",
            "available": True,
            "tenants": [
                {
                    "tenant_id": "tnt_1",
                    "tenant_name": "Org",
                    "eligible": True,
                    "endpoint": None,
                    "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
                }
            ],
            "consent": {"version": "bridge_consent_v1", "text": "the served text"},
        },
        from_cache=False,
        fetched_at=1.0,
        checked_at=1.0,
        error=None,
        account_id="acc_1",
        # The served consent is only displayable when it IS the accepted
        # accession (review round 1, R3): the fixture's payload and its
        # accepted record agree, as a real answer's do after acceptance.
        texts={
            "consent": AcceptedText(
                version="bridge_consent_v1",
                text="the served text",
                sha256="x",
                accepted_at=1.0,
            )
        },
    )
    base.update(overrides)
    return StatusCheck(**base)


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    FakeClient.behavior = {}
    FakeClient.instances = []
    discipline = FakeDiscipline(_status_check())
    monkeypatch.setattr(tb_client, "TeamBridgeClient", FakeClient)
    monkeypatch.setattr(
        tb_client,
        "get_status_discipline",
        lambda config_dir=None, base_url=None: discipline,
    )
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.include_router(desktop_team_bridge.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost", headers=AUTH
    ) as client:
        yield client, root, discipline


async def test_status_composes_state_local_config_consent_and_the_screening_receipt(api):
    client, _root, discipline = api
    response = await client.get("/v1/desktop/team-bridge/status?boundary=app_start")
    assert response.status_code == 200
    result = response.json()["result"]

    assert result["state"] == STATE_READY
    assert result["boundary"] == "app_start"
    assert result["from_cache"] is False
    # The §5.6 reading the renderer binds to (computed once, backend-side), and
    # the read family's availability receipt NEXT TO it (spec M-11 / review R-4):
    # a renderer must be able to read which read operations exist before it calls
    # one, rather than probing for them.
    assert result["capability"] == 1
    assert result["availability"] == {
        "metadata_inbox": True,
        # M1a: the detail read and the list excerpts EXIST on this build, and
        # both answer only through the shared admission predicate.
        "screened_detail": True,
        "read_ack": False,
        "local_events": False,
    }
    assert result["surface_state"] == "B"
    assert result["tenant_id"] == "tnt_1"
    assert result["visibility"] == {
        "settings": True,
        "inbox": False,
        "account_door": False,
        "onboarding": True,
    }
    assert result["read_only"] is False
    assert result["can_receive"] is False
    assert result["can_decide"] is False
    # The captured-account guard receipt (R6) and the refused-slot record (R3).
    assert result["consent_account_guard"] is True
    assert result["state_note"] is None
    assert result["texts_rejected"] == []
    # The local keys the renderer composes its gate from.
    assert result["local"] == {"enabled": False, "onboarding": "", "session": ""}
    # The consent renders exactly as served, with its source named.
    assert result["consent"] == {
        "version": "bridge_consent_v1",
        "text": "the served text",
        "source": "served",
    }
    # Screening is an invariant receipt, never a setting (§8, §13.9).
    assert result["screening"] == {"required": True}
    assert result["error"] is None
    assert discipline.boundaries == ["app_start"]


async def test_status_reports_signed_out_as_a_state_not_an_error(api, monkeypatch):
    client, _root, discipline = api
    discipline.check_result = _status_check(
        state=STATE_SIGNED_OUT, payload=None, account_id=None, fetched_at=None
    )
    response = await client.get("/v1/desktop/team-bridge/status")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["state"] == STATE_SIGNED_OUT
    assert result["consent"] is None


async def test_status_carries_a_refusal_as_data(api):
    client, _root, discipline = api
    discipline.check_result = _status_check(
        state="unavailable",
        error=BridgeRefusal("messaging_unavailable", "not launched", http_status=503),
    )
    response = await client.get("/v1/desktop/team-bridge/status")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["state"] == "unavailable"
    assert result["error"]["code"] == "messaging_unavailable"


async def test_settings_get_passes_the_payload_through(api):
    client, _root, _discipline = api
    payload = {"endpoint": None, "settings": {"tenant_id": "tnt_1", "eligible": True}}
    FakeClient.behavior["settings"] = payload
    response = await client.get("/v1/desktop/team-bridge/settings?tenant_id=tnt_1")
    assert response.status_code == 200
    assert response.json()["result"] == payload


async def test_sign_in_required_maps_to_a_401_with_its_own_code(api):
    client, _root, _discipline = api
    FakeClient.behavior["settings"] = BridgeSignInRequired(
        "Sign in to Radient to use agent messages."
    )
    response = await client.get("/v1/desktop/team-bridge/settings")
    assert response.status_code == 401
    assert response.json()["detail"] == {
        "code": "bridge_signin_required",
        "message": "Sign in to Radient to use agent messages.",
    }


async def test_put_settings_requires_state_in_the_body(api):
    client, _root, _discipline = api
    response = await client.put("/v1/desktop/team-bridge/settings", json={"tenant_id": "tnt_1"})
    assert response.status_code == 422


async def test_put_settings_requires_the_captured_account_guard(api):
    """R6: an unguarded settings write is not a shape this plane accepts.

    Without the captured account the dialog-replay guard is impossible to
    enforce, so the desktop body REFUSES the request outright (422, empty
    body semantics) and no bridge client is ever constructed.
    """

    client, _root, _discipline = api
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused"},
    )
    assert response.status_code == 422
    assert FakeClient.instances == []


async def test_put_allowlist_requires_the_captured_account_guard(api):
    client, _root, _discipline = api
    response = await client.put(
        "/v1/desktop/team-bridge/allowlist",
        json={"entries": []},
    )
    assert response.status_code == 422
    assert FakeClient.instances == []


async def test_a_stale_captured_account_maps_to_account_mismatch(api):
    """R6/Q1: the client's guard refusal surfaces as a refresh-status signal."""

    client, _root, discipline = api
    FakeClient.behavior["put_settings"] = BridgeRefusal(
        "account_mismatch", "prepared for a different account", http_status=409
    )
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused", "expected_account_id": "acc_a"},
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "account_mismatch"
    # The staleness signal: re-read status before the user tries again.
    assert detail["refresh_status"] is True
    assert discipline.invalidated == 1


async def test_the_captured_account_reaches_the_client_verbatim(api):
    client, _root, _discipline = api
    FakeClient.behavior["put_settings"] = {"endpoint": None}
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused", "expected_account_id": "acc_7"},
    )
    assert response.status_code == 200
    assert FakeClient.behavior["last_put_kwargs"]["expected_account_id"] == "acc_7"


async def test_put_settings_local_precondition_failures_are_422_client_bugs(api):
    client, _root, _discipline = api
    FakeClient.behavior["put_settings"] = ValueError(
        "a pause or opt-out must be sent alone (RFC §5.2)"
    )
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused", "label": "My agent", "expected_account_id": "acc_1"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"


async def test_put_settings_consent_conflict_flags_refresh_and_invalidates_the_cache(api):
    client, _root, discipline = api
    FakeClient.behavior["put_settings"] = BridgeRefusal(
        "consent_version_unknown", "the consent version moved", http_status=409
    )
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={
            "state": "active",
            "consent_version": "bridge_consent_v1",
            "expected_account_id": "acc_1",
        },
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "consent_version_unknown"
    # §5.2: refresh status and re-render — the cached consent must not be served.
    assert detail["refresh_status"] is True
    assert discipline.invalidated == 1


async def test_put_settings_refusal_carries_details_and_retry_after(api):
    client, _root, _discipline = api
    FakeClient.behavior["put_settings"] = BridgeRefusal(
        "rate_limited",
        "slow down",
        http_status=429,
        details={"retry_after_seconds": 7},
        retry_after_seconds=7,
    )
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused", "expected_account_id": "acc_1"},
    )
    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["retry_after_seconds"] == 7
    assert detail["details"] == {"retry_after_seconds": 7}


async def test_allowlist_get_composes_entries_and_the_view(api):
    client, _root, _discipline = api
    FakeClient.behavior["allowlist"] = {
        "allowlist": {
            "tenant_id": "tnt_1",
            "entries": [
                {"direction": "send", "effect": "allow", "target": {"endpoint_id": "bep_1"}}
            ],
        }
    }
    response = await client.get("/v1/desktop/team-bridge/allowlist")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["allowlist"]["tenant_id"] == "tnt_1"
    assert result["view"]["directions"]["send"] == {
        "mode": "listed_only",
        "allow": 1,
        "deny": 0,
    }
    assert result["view"]["cues"]["removing_allow_does_not_stop_open_thread"] is True


async def test_put_allowlist_rejects_more_than_two_hundred_entries(api):
    client, _root, _discipline = api
    entries = [
        {"direction": "send", "effect": "allow", "target": {"endpoint_id": f"bep_{i}"}}
        for i in range(201)
    ]
    response = await client.put(
        "/v1/desktop/team-bridge/allowlist",
        json={"entries": entries, "expected_account_id": "acc_1"},
    )
    assert response.status_code == 422


async def test_put_allowlist_forwards_the_replace_set(api):
    client, _root, _discipline = api
    entry = {"direction": "receive", "effect": "deny", "target": {"account_id": "acc_9"}}
    FakeClient.behavior["put_allowlist"] = {"allowlist": {"tenant_id": "tnt_1", "entries": [entry]}}
    response = await client.put(
        "/v1/desktop/team-bridge/allowlist",
        json={"entries": [entry], "tenant_id": "tnt_1", "expected_account_id": "acc_1"},
    )
    assert response.status_code == 200
    assert FakeClient.behavior["last_entries"] == [entry]
    result = response.json()["result"]
    assert result["view"]["directions"]["receive"]["mode"] == "class_default"


async def test_every_route_requires_the_desktop_bearer(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setattr(tb_client, "TeamBridgeClient", FakeClient)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.include_router(desktop_team_bridge.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer wrong-token"},
    ) as client:
        for method, path in [
            ("GET", "/v1/desktop/team-bridge/status"),
            ("GET", "/v1/desktop/team-bridge/settings"),
            ("PUT", "/v1/desktop/team-bridge/settings"),
            ("GET", "/v1/desktop/team-bridge/allowlist"),
            ("PUT", "/v1/desktop/team-bridge/allowlist"),
        ]:
            response = await client.request(method, path, json={"state": "paused"})
            assert response.status_code == 401, (method, path)


# ---------------------------------------------------------------------------
# The retained consumer (review round 3, Q6): the STATUS route over ONE
# server-style manager must see an EXTERNAL process's writes on the NEXT
# request. QA measured the defect: with the same retained manager, a fresh
# reader saw the new pair while this route kept returning the stale one until
# that very manager was reloaded (masked in production by the general
# settings route's explicit reload, which this test never calls).
# ---------------------------------------------------------------------------


def _external(lines: str) -> None:
    """Run a REAL separate process against the isolated config; fail loudly."""

    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    result = subprocess.run(
        [sys.executable, "-c", lines],
        env=env,
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest_asyncio.fixture
async def retained_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One retained manager == the dependency identity, with the REAL route stack.

    No FakeClient and no discipline stub: the isolated config carries NO
    Radient credentials, so the real discipline answers ``signed_out`` with
    zero wire calls while the LOCAL block still renders from the config keys —
    exactly the surface this regression is about. Mounted ASGI WITHOUT the
    app's lifespan: the defect under test is the retained manager's lifetime,
    not a startup hook, and that scope is stated rather than implied (this is
    not a full-daemon claim).
    """

    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)  # THE retained manager
    app.include_router(desktop_team_bridge.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost", headers=AUTH
    ) as client:
        yield client, root, app


async def test_status_reads_hot_writes_through_a_retained_manager(retained_api):
    """Q6: the NEXT status request sees an external write — no reload, no GET.

    Deliberately no fiction: no fixture poll, no manager replacement, no
    intervening general-settings GET (whose explicit reload would mask the
    failure). The response bodies below are the HTTP layer's real output.
    """

    client, root, app = retained_api
    manager = app.state.config_manager

    first = await client.get("/v1/desktop/team-bridge/status?boundary=app_start")
    assert first.status_code == 200
    first_body = first.json()["result"]
    assert first_body["state"] == STATE_SIGNED_OUT
    assert first_body["local"] == {"enabled": False, "onboarding": "", "session": ""}

    # A real separate process writes the registered keys through settings_io.
    _external(
        "import pathlib\n"
        "from local_operator import settings_io\n"
        "from local_operator.config import ConfigManager\n"
        f"cm = ConfigManager(pathlib.Path({str(root)!r}))\n"
        "settings_io.write_setting(cm, settings_io.BY_KEY['team_bridge.enabled'], True)\n"
        "settings_io.write_setting(cm, settings_io.BY_KEY['team_bridge.session'],"
        " 'synthetic-r3-live-session')\n"
    )
    second = await client.get("/v1/desktop/team-bridge/status?boundary=surface_open")
    second_local = second.json()["result"]["local"]
    assert second_local["enabled"] is True
    assert second_local["session"] == "synthetic-r3-live-session"
    assert app.state.config_manager is manager  # same dependency identity

    # Disable + clear converge on the next request as well.
    _external(
        "import pathlib\n"
        "from local_operator import settings_io\n"
        "from local_operator.config import ConfigManager\n"
        f"cm = ConfigManager(pathlib.Path({str(root)!r}))\n"
        "settings_io.write_setting(cm, settings_io.BY_KEY['team_bridge.enabled'], False)\n"
        "settings_io.write_setting(cm, settings_io.BY_KEY['team_bridge.session'], '')\n"
    )
    third = await client.get("/v1/desktop/team-bridge/status?boundary=surface_open")
    assert third.json()["result"]["local"] == {
        "enabled": False,
        "onboarding": "",
        "session": "",
    }

    # An onboarding-only document edit (a key settings_io cannot write — it is
    # deliberately unregistered) is adopted even though NO registered key
    # moved: the watcher swaps its snapshot while reporting no change.
    _external(
        "import pathlib, yaml\n"
        f"p = pathlib.Path({str(root)!r}, 'config.yml')\n"
        "doc = yaml.safe_load(p.read_text()) or {}\n"
        "doc.setdefault('values', {}).setdefault('team_bridge', {})['onboarding'] = 'seen'\n"
        "p.write_text(yaml.safe_dump(doc))\n"
    )
    fourth = await client.get("/v1/desktop/team-bridge/status?boundary=window_focus")
    assert fourth.json()["result"]["local"]["onboarding"] == "seen"

    # A malformed document keeps the LAST-GOOD snapshot: bytes untouched, no
    # ``.bad`` backup, values unchanged.
    malformed = "::: not yaml :::\n"
    _external(
        "import pathlib\n" f"pathlib.Path({str(root)!r}, 'config.yml').write_text({malformed!r})\n"
    )
    malformed_bytes = (root / "config.yml").read_bytes()
    fifth = await client.get("/v1/desktop/team-bridge/status?boundary=window_focus")
    fifth_local = fifth.json()["result"]["local"]
    assert fifth_local == {"enabled": False, "onboarding": "seen", "session": ""}
    # The unreadable document is left EXACTLY as written: no repair, no
    # rewrite, no ``config.yml.bad.<ts>`` moved aside (the last-good snapshot
    # is kept instead — the write guard's behaviour, not ``_load_config``'s).
    assert (root / "config.yml").read_bytes() == malformed_bytes
    assert list(root.glob("config.yml.bad*")) == []

    # A restored valid document advances.
    _external(
        "import pathlib, yaml\n"
        f"p = pathlib.Path({str(root)!r}, 'config.yml')\n"
        "p.write_text(yaml.safe_dump({'values': {'team_bridge':"
        " {'enabled': True, 'session': 'synthetic-r3-restored'}}}))\n"
    )
    sixth = await client.get("/v1/desktop/team-bridge/status?boundary=surface_open")
    sixth_local = sixth.json()["result"]["local"]
    assert sixth_local["enabled"] is True
    assert sixth_local["session"] == "synthetic-r3-restored"
    assert app.state.config_manager is manager


# -- the READ slice: the inbox read (metadata only, zero local writes) ---------

INBOX = "/v1/desktop/team-bridge/inbox"
HUB = "https://hub.example"


def _access(account_id: str = "acc_1", credential_id: int = 7) -> OAuthAccess:
    return OAuthAccess(
        access_token="synthetic-bearer",
        credential_id=credential_id,
        account_id=account_id,
        kind="oauth",
    )


def _owner(*, tenant_id: str = "tnt_1", generation: int = 5) -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_1",
        tenant_id=tenant_id,
        endpoint_id="ep_1",
        canonical_base_url=HUB,
        enrollment_generation=generation,
    )


# --- M1a shared fixtures: the wire item and the proof that could admit it ---


def _wire_item(
    *,
    message_id: str = "bmsg_1",
    body: str = "could you approve the Q3 spend for the enrichment pod?",
    state: str = "read",
    type_: str = "note",
    sender_name: str | None = "Priya Raman",
    body_hash: str | None = None,
    private_reason: str | None = None,
) -> dict[str, Any]:
    """ONE upstream item whose OWN hash matches — the shape the hub sends.

    Every field the predicate reads is present, so a case that denies content can
    only be denying it because of the term that case breaks, not because the
    fixture was thin. ``private_reason``/body are the sentinels the privacy cells
    put ON THE WIRE: nothing may ever return them verbatim.
    """

    text = body.encode("utf-8")
    message: dict[str, Any] = {
        "id": message_id,
        "thread_id": "bth_1",
        "tenant_id": "tnt_1",
        "type": type_,
        "seq": 4,
        "origin": "ep_9",
        "body": {"text": body},
        "body_hash": body_hash or hashlib.sha256(text).hexdigest(),
        "recipient": {"account_id": "acc_1", "endpoint_id": "ep_1"},
    }
    if private_reason is not None:
        message["private_reason"] = private_reason
    sender: dict[str, Any] = {"account_id": "acc_9", "endpoint_id": "ep_9"}
    if sender_name is not None:
        sender["name"] = sender_name
    return {
        "message": message,
        "delivery": {"state": state, "state_at": None},
        "thread": {"id": "bth_1", "status": "open", "tenant_id": "tnt_1"},
        "sender": sender,
    }


def _plant_proof(
    root: Path,
    owner: OwnerSnapshot,
    item: Mapping[str, Any],
    **overrides: Any,
) -> None:
    """Plant the policy + receipt + CONFIRMED read intent that could admit ``item``.

    The receipt's two hashes are DERIVED from the item with the product's own
    validator, so the fixture cannot accidentally mint a tuple the pipeline would
    never have written; each override then breaks exactly ONE term, which is what
    makes a refusal cell discriminating rather than generic.
    """

    envelope, code = validate_envelope(owner, item)
    assert code is None and envelope is not None, code
    # The PUBLISHED policy and the RECEIPT can disagree in these three fields, and
    # that disagreement is exactly what the currency terms exist to catch — so a
    # case may set them apart rather than only moving the pair together.
    template: str = overrides.pop("template", "ab" * 32)
    model: str = overrides.pop("model", "m-1")
    revision: str = overrides.pop("revision", "r-1")
    policy_template: str = overrides.pop("policy_template", template)
    policy_model: str = overrides.pop("policy_model", model)
    policy_revision: str = overrides.pop("policy_revision", revision)
    verdict: str = overrides.pop("verdict", "allow")
    categories: tuple[str, ...] = overrides.pop("categories", ())
    generation: int = overrides.pop("generation", owner.enrollment_generation)
    intent: str | None = overrides.pop("intent", "confirmed")
    receipt_body: str = overrides.pop("receipt_body", envelope.body_sha256)
    receipt_envelope: str = overrides.pop("receipt_envelope", envelope.envelope_digest)
    policy: bool = overrides.pop("policy", True)
    with_receipt: bool = overrides.pop("receipt", True)
    epoch_offset: int = overrides.pop("epoch_offset", 0)
    categories_text: str | None = overrides.pop("categories_text", None)
    hold: str | None = overrides.pop("hold", None)
    intent_body: str = overrides.pop("intent_body", receipt_body)
    assert not overrides, overrides

    store = Store(root)
    epoch = 0
    if policy:
        epoch = store.publish_screening_policy(
            owner,
            version="bridge-screen-v1",
            template_sha256=policy_template,
            default_model=policy_model,
            revision=policy_revision,
        )
    if with_receipt:
        with sqlite3.connect(state_db_path(root)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO screening_receipts (owner_key, message_id, body_sha256, "
                "template_sha256, prompt_version, model, verdict, categories, envelope_digest, "
                "admission_revision, policy_epoch, creator_generation, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _owner_key(owner),
                    str(item["message"]["id"]),
                    receipt_body,
                    template,
                    "bridge-screen-v1",
                    model,
                    verdict,
                    (
                        categories_text
                        if categories_text is not None
                        else json.dumps(list(categories))
                    ),
                    receipt_envelope,
                    revision,
                    epoch + epoch_offset,
                    generation,
                    1_700_000_000.0,
                ),
            )
    if hold is not None:
        with sqlite3.connect(state_db_path(root)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO screening_holds "
                "(owner_key, message_id, hold_code, retry_at, created_at) VALUES (?,?,?,?,?)",
                (_owner_key(owner), str(item["message"]["id"]), hold, None, 1_700_000_000.0),
            )
    if intent is not None:
        # The intent's OWN tuple rides its own receipt object: a case that wants a
        # tuple nobody could have written passes ``intent_body``.
        planted = ScreeningReceipt(
            verdict="allow",
            model=model,
            prompt_version="bridge-screen-v1",
            template_sha256=template,
            body_sha256=intent_body,
            categories=categories,
        )
        intent_id = store.record_ack_intent(
            owner, str(item["message"]["id"]), "read", planted, "r" * 36
        )
        if intent == "confirmed":
            store.mark_ack_result(owner, intent_id, state="confirmed", error_code=None)
        elif intent == "conflict":
            store.mark_ack_result(owner, intent_id, state="conflict", error_code="conflict")


def _enroll(root: Path, *, enabled: bool = True) -> OwnerSnapshot:
    """Commit the owner + consent receipt H4 commits, and set the local switch.

    The switch is the DOCUMENT the running daemon reads (`values.team_bridge.
    enabled`), written the way the settings page writes it — the read path reads
    the file from disk, never a long-lived manager, so a fixture that set an
    in-memory value would test nothing.
    """
    owner = _owner()
    Store(root).commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"team_bridge": {"enabled": enabled}}})
    )
    return owner


def _query(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"expected_account_id": "acc_1", "tenant_id": "tnt_1"}
    base.update(overrides)
    return base


def _item(
    message_id: str = "bmsg_1",
    *,
    state: str = "unread",
    tenant_id: str = "tnt_1",
    recipient_account: str = "acc_1",
    recipient_endpoint: str | None = "ep_1",
    body: Any = None,
    private_reason: Any = None,
) -> dict[str, Any]:
    """One upstream item, in the frozen nested shape (``_project_item`` reads it).

    ``body``/``private_reason`` are the sentinels the privacy cells put ON THE
    WIRE: the projection never reads them, and the route must not either.
    """
    message: dict[str, Any] = {
        "id": message_id,
        "thread_id": "bth_1",
        "tenant_id": tenant_id,
        "type": "note",
        "seq": 4,
        "origin": "ep_9",
        "body_hash": "a" * 64,
        "recipient": {"account_id": recipient_account, "endpoint_id": recipient_endpoint},
    }
    if body is not None:
        message["body"] = body
    if private_reason is not None:
        message["private_reason"] = private_reason
    return {
        "message": message,
        "delivery": {"state": state, "state_at": None},
        "thread": {"id": "bth_1", "status": "open"},
        "sender": {"account_id": "acc_9", "endpoint_id": "ep_9"},
    }


def _page(*items: dict[str, Any], next_cursor: str | None = None) -> dict[str, Any]:
    return {
        "items": list(items),
        "next_cursor": next_cursor,
        "counts": {"unread": len(items), "requests_pending": 0, "blocked": 0},
    }


def _digest(root: Path) -> tuple[list[str], str]:
    """(new file names under the root, a content digest of every SQLite table).

    The zero-write proof, and it compares STATE rather than page bytes: SQLite's
    own ``CREATE TABLE IF NOT EXISTS`` can touch a database file without changing
    a row, so a proof built on mtimes would be a proof about SQLite. SQLite's
    WAL companions are excluded from the name list for the same reason — a reader
    may need them, and neither is this surface's write. An ack intent, a merged
    ref or a moved connector checkpoint WOULD show up here, in either half.

    BOTH database suffixes are hashed (review R-1). The bridge store is
    ``state.sqlite3``, not ``*.db`` — so the original ``rglob("*.db")`` matched
    NOTHING and the "state unchanged" half of every cell compared an empty digest,
    which is how a store this surface created went unseen.
    """
    names = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(("-wal", "-shm"))
    )
    digest = hashlib.sha256()
    for db in sorted({*root.rglob("*.db"), *root.rglob("*.sqlite3")}):
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            tables = [
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            for table in tables:
                digest.update(table.encode("utf-8"))
                for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1"):  # noqa: S608
                    digest.update(repr(row).encode("utf-8"))
        finally:
            connection.close()
    return names, digest.hexdigest()


def _seed_allow_receipt(root: Path, message_id: str) -> None:
    """Write ONE native screening receipt row into the committed store.

    Deliberately raw SQL: the receipt WRITER is the screening pipeline's own
    (``Store.finish_screening``, fenced on an attempt row, a policy fingerprint
    and the published epoch), and this cell's question is narrower — what a bare
    receipt ROW may make a read surface claim (review R-3 / M-12a). The owner key
    is read back out of ``enrollments`` rather than rebuilt here, so the seed
    cannot drift from the store's own encoding. Same seeding style as
    ``tests/unit/team_bridge/test_h5_envelope_negatives.py``.
    """

    with sqlite3.connect(state_db_path(root)) as conn:
        row = conn.execute("SELECT owner_key FROM enrollments WHERE state = 'committed'").fetchone()
        assert row is not None
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, "
            "envelope_digest, admission_revision, policy_epoch, creator_generation, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(row[0]),
                message_id,
                "a" * 64,
                "b" * 64,
                "prompt-v1",
                "model-a",
                "allow",
                '["spam"]',
                "c" * 64,
                "adm-1",
                1,
                5,
                0.0,
            ),
        )


async def test_the_inbox_read_needs_the_desktop_bearer(api):
    """The plane's gate first: a wrong or absent bearer never reaches the read.

    The fixture's client carries the correct bearer, and httpx has no per-request
    "remove this header", so this cell edits the per-test client's default header
    — the same object the gate reads on each request.
    """
    client, root, _discipline = api
    _enroll(root)
    original = client.headers["Authorization"]

    client.headers["Authorization"] = "Bearer not-the-token"
    wrong = await client.get(INBOX, params=_query())
    del client.headers["Authorization"]
    absent = await client.get(INBOX, params=_query())
    client.headers["Authorization"] = original

    assert wrong.status_code == 401
    assert absent.status_code == 401
    # No client, no wire call, no store work: the gate is a dependency.
    assert FakeClient.instances == []
    assert FakeClient.behavior.get("last_inbox_kwargs") is None


async def test_a_read_that_names_a_bad_id_is_refused_before_any_upstream_path(api):
    """S-RSA-5: grammar on the ids, BEFORE the transport can build a path.

    The upstream interpolates ids into ``/messaging/messages/{id}`` and folds
    them into query strings, so ``..`` or ``?`` in one rewrites the request; the
    probe that found this built ``/v1/messaging`` from ``..``. The rejected VALUE
    must not ride the refusal back either — the sentence is fixed.
    """
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item())

    for bad in (
        _query(thread_id="../etc/passwd"),
        _query(sender_account_id="acc_1?tenant_id=tnt_other"),
        _query(after="x?y=1"),
        _query(state="screened"),
        _query(expected_account_id="acc_1/../acc_2"),
        # A trailing newline (review R-5): ``$`` matches before it, so only a
        # ``fullmatch`` validator refuses this value at the door.
        _query(thread_id="bth_1\n"),
        _query(expected_account_id="acc_1\n"),
    ):
        response = await client.get(INBOX, params=bad)
        assert response.status_code == 422, bad
        detail = response.json()["detail"]
        assert detail["code"] == "invalid_request"
        assert detail["message"] == desktop_team_bridge.INVALID_READ_QUERY
        assert str(bad) not in response.text

    # The grammar runs before the client exists: zero upstream path, zero calls.
    assert FakeClient.instances == []
    assert FakeClient.behavior.get("last_inbox_kwargs") is None


async def test_the_inbox_read_refuses_a_wrong_account_with_zero_upstream_and_zero_writes(api):
    """The captured-account receipt, on a READ: A's inbox is never B's answer."""
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access("acc_other")
    FakeClient.behavior["inbox"] = _page(_item())
    before = _digest(root)

    response = await client.get(INBOX, params=_query())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "account_mismatch"
    assert (
        FakeClient.behavior.get("last_inbox_kwargs") is None
    ), "no wire call for a refused account"
    assert _digest(root) == before


async def test_the_inbox_read_answers_metadata_only_and_writes_nothing(api):
    """The ordinary page: ids, states, coarse codes — and not one local write."""
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()
    # The wire carries a body and a private reason; neither may reach the answer.
    FakeClient.behavior["inbox"] = _page(
        _item("bmsg_1", body={"text": "BODY-SENTINEL"}, private_reason="REASON-SENTINEL"),
        _item("bmsg_2", state="read"),
        next_cursor="MjU=",
    )
    before = _digest(root)

    response = await client.get(INBOX, params=_query())

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["mode"] == "receive"
    assert result["read_only"] is False
    assert result["history_reason"] is None
    assert result["read_capability"] == 1
    assert result["account_id"] == "acc_1"
    assert result["tenant_id"] == "tnt_1"
    assert result["owner"]["hub_origin"] == HUB
    assert result["next_cursor"] == "MjU="
    assert result["counts"] == {"items": 2, "unread": 1, "held": 0, "blocked": 0}
    assert result["body_included"] is False
    assert result["screening_detail_included"] is False
    assert result["local_events"] is False
    # The row is the ALLOWLIST, exactly — no field may appear beside it.
    assert set(result["items"][0]) == {
        "message_id",
        "thread_id",
        "message_type",
        "origin",
        "seq",
        "delivery",
        "sender",
        "recipient",
        "request",
        "screening",
        # The ONE field that may carry third-party text, and it is null unless
        # the shared predicate admitted these bytes (M1a).
        "content",
    }
    assert result["items"][0]["content"] is None
    assert result["items"][0]["message_id"] == "bmsg_1"
    assert result["items"][0]["delivery"] == {"state": "unread", "state_at": None}
    assert result["items"][0]["screening"] == {
        "state": "unreviewed",
        "hold_code": None,
        "retry_at": None,
    }
    assert "BODY-SENTINEL" not in response.text
    assert "REASON-SENTINEL" not in response.text
    # The wire call carried the caller's scope, never a widened one.
    wire = FakeClient.behavior["last_inbox_kwargs"]
    assert wire["tenant_id"] == "tnt_1"
    assert wire["limit"] == 50
    assert _digest(root) == before, "a read moved local state"


async def test_the_inbox_read_reports_a_hold_as_a_coarse_code_and_never_its_text(api):
    """A connector-written hold can quote provider prose; the row reports a CODE."""
    client, root, _discipline = api
    owner = _enroll(root)
    Store(root).persist_hold(
        owner, "bmsg_2", "provider_error: upstream said HOLD-PROSE", retry_at=123.0
    )
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item("bmsg_2"))

    response = await client.get(INBOX, params=_query())

    assert response.status_code == 200
    row = response.json()["result"]["items"][0]
    assert row["screening"]["state"] == "held"
    assert row["screening"]["hold_code"] == "unknown_hold"
    assert row["screening"]["retry_at"] == 123.0
    assert "HOLD-PROSE" not in response.text
    assert response.json()["result"]["counts"]["held"] == 1


async def test_an_allow_receipt_does_not_make_a_row_allowed(api):
    """M-12a / review R-3: a bare receipt is not admission, so the row stays unreviewed."""
    client, root, _discipline = api
    _enroll(root)
    _seed_allow_receipt(root, "bmsg_1")
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item("bmsg_1"))

    response = await client.get(INBOX, params=_query())

    assert response.status_code == 200
    assert response.json()["result"]["items"][0]["screening"] == {
        "state": "unreviewed",
        "hold_code": None,
        "retry_at": None,
    }


async def test_the_read_refuses_a_wire_that_cannot_name_the_committed_hub(api, monkeypatch):
    """Q3: the destination guard is load-bearing — a mutated base_url must refuse.

    QA round 1 measured that DELETING this guard left the whole suite green: the
    only thing that exercised it was an import-order accident. Both refusals fire
    BEFORE the wire is called.
    """
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item())

    monkeypatch.setattr(FakeClient, "base_url", "https://evil.example")
    mismatched = await client.get(INBOX, params=_query())
    assert mismatched.status_code == 409
    assert mismatched.json()["detail"]["code"] == "destination_mismatch"
    assert FakeClient.behavior.get("last_inbox_kwargs") is None
    assert "evil.example" not in mismatched.text

    monkeypatch.setattr(FakeClient, "base_url", "")
    silent = await client.get(INBOX, params=_query())
    assert silent.status_code == 502
    assert silent.json()["detail"]["code"] == "no_destination_evidence"
    assert FakeClient.behavior.get("last_inbox_kwargs") is None


async def test_a_read_refuses_when_the_owner_moves_while_the_wire_answers(api):
    """Q4: the post-await ``_recheck`` is load-bearing (``owner_changed``).

    The store moves INSIDE the wire call — after the standing check, before the
    page is projected — which is the window the re-check exists for. Measured by
    QA round 1: removing it left the whole suite green.
    """
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item())

    def move_owner() -> None:
        Store(root).commit_enrollment(
            _owner(generation=6), OwnerReceipt("bridge_consent_v1", "ref-2")
        )

    FakeClient.behavior["during_inbox"] = move_owner
    response = await client.get(INBOX, params=_query())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "owner_changed"


async def test_a_read_refuses_a_query_name_it_does_not_accept(api):
    """Q5: an unknown query name is REFUSED, never silently dropped.

    FastAPI drops unknown query names, so a caller's typo (``threadId``) would
    otherwise read a WIDER page than the one they asked for — the model's
    ``extra=forbid`` never sees the query string.
    """
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()

    response = await client.get(INBOX, params={**_query(), "threadId": "bth_1"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
    assert "threadId" not in response.text
    assert FakeClient.instances == []


async def test_a_read_over_a_bound_field_refuses_without_echoing_the_value(api):
    """R-6: every bound lives in the model, so no refusal echoes the input."""
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()

    oversize = "A" * 600
    for bad in (_query(after=oversize), _query(limit="SECRET-LIMIT"), _query(state="x" * 40)):
        response = await client.get(INBOX, params=bad)
        assert response.status_code == 422, bad
        assert response.json()["detail"]["code"] == "invalid_request"
        assert "SECRET" not in response.text
        assert "AAAA" not in response.text
    assert FakeClient.instances == []


async def test_a_damaged_store_is_refused_typed_and_left_alone(api):
    """M2-1: a store file that is not a database refuses typed, and is not repaired.

    The read-only accessors map any ``sqlite3.Error`` to ``state_unreadable``
    (review R-7), and until this cell that mapping had no test of its own: the
    reviewer's throw-away probe was the only thing that had ever reached it.
    """
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item())

    damaged = b"this is not a database" * 32
    store_file = state_db_path(root)
    store_file.write_bytes(damaged)

    response = await client.get(INBOX, params=_query())

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "state_unreadable"
    assert detail["message"] == desktop_team_bridge.STATE_UNREADABLE
    assert store_file.read_bytes() == damaged, "the read repaired the store it refused"
    assert not list(root.rglob("state.sqlite3-wal"))
    assert FakeClient.behavior.get("last_inbox_kwargs") is None


async def test_the_read_path_never_reaches_a_write_capable_store_accessor(api, monkeypatch):
    """Q6 / R-1: swapping a writable accessor into the read path must fail here.

    The review's mutation put the writable ``Store.active_owner`` back inside
    ``_receive_fence`` and the whole suite stayed green: on that path the store
    already exists, so nothing observable changes and the main-file bytes are
    identical either way. No state-based assertion can pin that, so this cell
    pins the STRUCTURE — every write-capable reader this slice replaced raises if
    the read path reaches it — and then drives both the ordinary page and the
    nothing-enrolled refusal through it.
    """
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item())

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the read path called a write-capable store accessor")

    for name in ("active_owner", "owner_consent_receipt", "screening_hold", "receipt_for_message"):
        monkeypatch.setattr(Store, name, forbidden)
    with pytest.raises(AssertionError):
        # The guard is live, not a silently failing patch of a renamed method.
        Store(root).active_owner()

    read = await client.get(INBOX, params=_query())
    assert read.status_code == 200
    assert read.json()["result"]["items"][0]["screening"]["state"] == "unreviewed"

    # With the store gone, the refusal path must not reach them either.
    state_db_path(root).unlink()
    absent = await client.get(INBOX, params=_query())
    assert absent.status_code == 409
    assert absent.json()["detail"]["code"] == "history_context_unavailable"


async def test_a_machine_with_the_local_switch_off_reads_retained_history_read_only(api):
    """Paused/disabled: the same participant may READ what already arrived."""
    client, root, _discipline = api
    _enroll(root, enabled=False)
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item())
    before = _digest(root)

    response = await client.get(INBOX, params=_query())

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["mode"] == "history"
    assert result["read_only"] is True
    assert result["history_reason"] == "bridge_off"
    assert result["items"][0]["message_id"] == "bmsg_1"
    assert _digest(root) == before


async def test_the_read_refuses_a_page_that_is_not_this_holders(api):
    """A foreign page refuses WHOLE: no partial render, no adoption."""
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()

    FakeClient.behavior["inbox"] = _page(_item("bmsg_1"), _item("bmsg_2", tenant_id="tnt_other"))
    foreign = await client.get(INBOX, params=_query())
    assert foreign.status_code == 409
    assert foreign.json()["detail"]["code"] == "wrong_tenant"
    assert "bmsg_1" not in foreign.text

    FakeClient.behavior["inbox"] = _page(_item("bmsg_1", recipient_endpoint="ep_other"))
    other_endpoint = await client.get(INBOX, params=_query())
    assert other_endpoint.status_code == 409
    assert other_endpoint.json()["detail"]["code"] == "wrong_recipient"
    assert "bmsg_1" not in other_endpoint.text

    # NO recipient endpoint is the same refusal: the connector's own binding
    # (``Store._check_binding``) requires the endpoint, so a row ``pull.inbox``
    # would drop must not render here either (round-1 review).
    before = _digest(root)
    FakeClient.behavior["inbox"] = _page(_item("bmsg_1", recipient_endpoint=None))
    unaddressed = await client.get(INBOX, params=_query())
    assert unaddressed.status_code == 409
    assert unaddressed.json()["detail"]["code"] == "wrong_recipient"
    assert "bmsg_1" not in unaddressed.text
    assert _digest(root) == before, "a refused read wrote local state"


async def test_the_read_refuses_a_machine_with_no_retained_namespace(api):
    """Nothing enrolled for this account: typed unavailable, never a guess."""
    client, root, _discipline = api
    (root / "config.yml").write_text(yaml.safe_dump({"values": {"team_bridge": {"enabled": True}}}))
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item())

    before = _digest(root)
    response = await client.get(INBOX, params=_query())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "history_context_unavailable"
    assert FakeClient.behavior.get("last_inbox_kwargs") is None
    # AND IT MADE NOTHING (review R-1/M-9): the store is opened READ-ONLY, so
    # "this machine holds nothing for you" must not leave a fresh,
    # DDL-initialised ``team_bridge/state.sqlite3`` behind.
    assert not (root / "team_bridge").exists(), "a refused read created the bridge store"
    assert not list(root.rglob("state.sqlite3"))
    assert _digest(root) == before


async def test_the_read_refuses_a_tenant_this_machine_did_not_enroll_with(api):
    """The named organization IS the scope; another one is not a filter."""
    client, root, _discipline = api
    _enroll(root)
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["inbox"] = _page(_item())

    response = await client.get(INBOX, params=_query(tenant_id="tnt_other"))

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_tenant"
    assert FakeClient.behavior.get("last_inbox_kwargs") is None


def test_the_read_slice_keeps_httpx_off_the_boot_path(tmp_path: Path) -> None:
    """`lop serve` imports the router; the client stack must stay in handlers."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CMUX_", "LOP_"))}
    env.update(HOME=str(tmp_path), LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / ".local-operator"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from local_operator.server.routes import desktop_team_bridge as m; "
            "paths = {str(r.path) for r in m.router.routes}; "
            "assert '/v1/desktop/team-bridge/inbox' in paths, paths; "
            "assert 'httpx' not in sys.modules, 'the read put httpx on the boot path'; "
            "print('read slice: lazy')",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "lazy" in result.stdout


def test_the_read_capability_is_advertised_without_moving_its_sibling() -> None:
    """Its OWN key (§7): reads are not a bump, and not a decision promise."""
    from local_operator.server.features import feature_flags

    flags = feature_flags()
    assert flags["team_bridge_read"] == 1
    assert flags["team_bridge"] == 1


# --- M1a: the single-message detail read -------------------------------------

DETAIL = "/v1/desktop/team-bridge/messages"


async def test_the_detail_read_answers_metadata_only_without_proof(api):
    """No local proof is the ORDINARY case, and it is not an error.

    A machine that has never screened this message has nothing that may admit its
    bytes, so the answer is the row plus ``content: null`` — a renderer draws the
    message and says it cannot show the text, instead of the route inventing
    either an error or a body.
    """

    client, root, _discipline = api
    FakeClient.behavior["access"] = _access()
    _enroll(root)
    FakeClient.behavior["message"] = _wire_item(
        private_reason="the model said the sender was pushy"
    )

    response = await client.get(f"{DETAIL}/bmsg_1", params=_query())

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["content"] is None
    assert result["item"]["content"] is None
    assert result["body_included"] is False
    assert result["screening_detail_included"] is False
    assert result["local_events"] is False
    assert result["item"]["message_id"] == "bmsg_1"
    assert result["item"]["screening"]["state"] == "unreviewed"
    # The wire's own private field never rides any answer.
    assert "pushy" not in response.text


async def test_the_detail_read_serves_the_body_only_on_the_whole_tuple(api):
    """The positive control beside the metadata cells: proof present, body served.

    The text is the fetched body, labelled as inert third-party content; the same
    predicate that refused the cells above is what let this one through, so the
    pair of them is what proves the predicate — not the assertion, and not a
    second code path that happens to return text.
    """

    client, root, _discipline = api
    FakeClient.behavior["access"] = _access()
    owner = _enroll(root)
    item = _wire_item()
    _plant_proof(root, owner, item)
    FakeClient.behavior["message"] = item

    response = await client.get(f"{DETAIL}/bmsg_1", params=_query())

    assert response.status_code == 200
    result = response.json()["result"]
    body = item["message"]["body"]["text"]
    assert result["content"]["text"] == body
    assert result["content"]["content_trust"] == "untrusted_external"
    assert result["content"]["content_format"] == "plain_text"
    assert result["content"]["truncated"] is False
    assert result["content"]["content_sha256"] == item["message"]["body_hash"]
    assert result["body_included"] is True
    assert result["screening_detail_included"] is True


@pytest.mark.parametrize(
    "bad_id",
    # The last-but-one value is the ENCODED newline: it is the one ``re.match``
    # would have accepted (``$`` matches before a trailing newline), so it is the
    # value that proves the route's ``fullmatch``. It rides percent-encoded
    # because a raw one cannot be put in a URL at all.
    ["bmsg_", "msg_1", "bmsg_1;x", "bmsg_1%0A", "bmsg_" + "a" * 65, "bmsg_1."],
)
async def test_a_message_id_outside_the_grammar_is_a_404_before_any_request(api, bad_id):
    """The id rides the PATH, so it is checked before the transport exists.

    ``FakeClient.instances == []`` is the load-bearing half: a 404 alone would not
    say whether the value reached the wire first. The trailing-newline value is
    the one ``re.match`` would have ACCEPTED (``$`` matches before it), which is
    why the route uses ``fullmatch``.
    """

    client, root, _discipline = api
    FakeClient.behavior["access"] = _access()
    _enroll(root)

    response = await client.get(f"{DETAIL}/{bad_id}", params=_query())

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "message_not_found"
    assert FakeClient.instances == [], "a malformed id reached the transport"


async def test_the_detail_read_refuses_a_query_name_it_does_not_accept(api):
    """An unknown name is refused, never dropped: a typo must not widen a read."""

    client, root, _discipline = api
    FakeClient.behavior["access"] = _access()
    _enroll(root)

    response = await client.get(f"{DETAIL}/bmsg_1", params={**_query(), "threadId": "bth_1"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
    assert "threadId" not in response.text, "the refused value was echoed back"
    assert FakeClient.instances == []


async def test_the_detail_read_refuses_a_missing_capture_and_a_foreign_tenant(api):
    """Both scope terms are required, and the org is the enrollment, not a filter."""

    client, root, _discipline = api
    FakeClient.behavior["access"] = _access()
    _enroll(root)

    missing = await client.get(f"{DETAIL}/bmsg_1", params={"tenant_id": "tnt_1"})
    assert missing.status_code == 422
    assert missing.json()["detail"]["code"] == "invalid_request"
    assert FakeClient.instances == [], "a query with no capture reached the transport"

    foreign = await client.get(f"{DETAIL}/bmsg_1", params={**_query(), "tenant_id": "tnt_2"})
    assert foreign.status_code == 422
    assert foreign.json()["detail"]["code"] == "invalid_tenant"
    # The wire object exists by then (§3 constructs it before the org check, the
    # same order the page uses); what must NOT exist is a request on it.
    assert FakeClient.behavior.get("last_message_kwargs") is None


async def test_the_detail_read_is_zero_write(api):
    """The same invariant the page has: state before == state after (M-9/M-10)."""

    client, root, _discipline = api
    FakeClient.behavior["access"] = _access()
    owner = _enroll(root)
    item = _wire_item()
    _plant_proof(root, owner, item)
    FakeClient.behavior["message"] = item

    before = _digest(root)
    response = await client.get(f"{DETAIL}/bmsg_1", params=_query())
    after = _digest(root)

    assert response.status_code == 200
    assert before == after, "the detail read wrote to the bridge store"


async def test_a_held_message_answers_no_body_and_no_reason(api):
    """A hold outranks the receipt, and the stored code is reported COARSE.

    ``ack_not_confirmed:pending`` is connector-written and outside the closed hold
    vocabulary, so it is reported as ``unknown_hold``: the raw string could quote
    provider text, and the private reason beside it must not appear either.
    """

    client, root, _discipline = api
    FakeClient.behavior["access"] = _access()
    owner = _enroll(root)
    item = _wire_item(private_reason="quarantined for a reason")
    _plant_proof(root, owner, item, hold="ack_not_confirmed:pending")
    FakeClient.behavior["message"] = item

    response = await client.get(f"{DETAIL}/bmsg_1", params=_query())

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["content"] is None
    assert result["item"]["screening"]["state"] == "held"
    assert result["item"]["screening"]["hold_code"] == "unknown_hold"
    assert "ack_not_confirmed" not in response.text
    assert "quarantined" not in response.text


async def test_a_re_enrollment_during_the_detail_fetch_denies_the_body(api):
    """The post-await fence is load-bearing on THIS path too.

    Measured against its mutation: deleting the detail's ``_recheck`` leaves this
    cell (and every other cell here) green, which is why the hook exists.
    """

    client, root, _discipline = api
    FakeClient.behavior["access"] = _access()
    owner = _enroll(root)
    item = _wire_item()
    _plant_proof(root, owner, item)
    FakeClient.behavior["message"] = item

    def move_owner() -> None:
        Store(root).commit_enrollment(
            _owner(generation=6), OwnerReceipt("bridge_consent_v1", "ref-2")
        )

    FakeClient.behavior["during_message"] = move_owner

    response = await client.get(f"{DETAIL}/bmsg_1", params=_query())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "owner_changed"


async def test_the_detail_read_refuses_a_machine_with_no_retained_namespace(api):
    """The detail's early refusal, and the store it must not leave behind.

    A machine with nothing enrolled for the signed-in account answers typed and
    unavailable — and it answers it WITHOUT creating the bridge store, which is
    the half a "409 either way" assertion would miss: ``state.sqlite3`` is
    ``mode=ro`` on an existing file or nothing at all, never a database this
    route made to explain its own answer.
    """

    client, root, _discipline = api
    (root / "config.yml").write_text(yaml.safe_dump({"values": {"team_bridge": {"enabled": True}}}))
    FakeClient.behavior["access"] = _access()
    FakeClient.behavior["message"] = _wire_item()

    before = _digest(root)
    response = await client.get(f"{DETAIL}/bmsg_1", params=_query())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "history_context_unavailable"
    assert FakeClient.behavior.get("last_message_kwargs") is None
    assert not (root / "team_bridge").exists(), "a refused detail read created the bridge store"
    assert not list(root.rglob("state.sqlite3"))
    assert _digest(root) == before

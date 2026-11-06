"""A same-identity re-login no longer leaves the Team Bridge pinned to a dead row.

The 2026-11-06 incident, reproduced from its real state: signing out of Radient
and back in as the SAME account deleted credential row 50 and inserted 136, the
enrollment in ``team_bridge/state.sqlite3`` stayed committed with
``credential_id=50``, and every bridge call (and every receiver iteration)
answered "sign in to Radient" to someone who was signed in.

Driven through the real paths: the receiver's ``run_forever`` with the real
``TeamBridgeClient.current_access`` as its access provider (only the clock is
simulated), ``auth_cli.run_login`` with only the IdP synthetic, and the real
host runtime + tool op over an ``httpx.MockTransport``. Every store is a temp
config root; nothing reaches the operator's live store.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.providers import auth_cli
from local_operator.providers.auth_store import AuthStore
from local_operator.team_bridge import host
from local_operator.team_bridge.client import (
    CONSENT_VERSION_V1,
    BridgeSignInRequired,
    TeamBridgeClient,
)
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.store import OwnerReceipt, Store, state_db_path
from local_operator.team_bridge.tool import execute_bridge
from local_operator.team_bridge.types import OwnerSnapshot
from local_operator.tunnels import config as tunnel_config
from tests.unit.team_bridge.test_connector_rearm import _SimulatedClock
from tests.unit.test_tunnel_pin_heal import (  # noqa: F401 — the tunnel's own fixtures
    _configure,
    _seed_gone_pin,
    _stub_the_identity_provider,
    sandbox,
)

ACCOUNT = "acct-operator"  # the tunnel tests' account, so one login heals both
HUB = DEFAULT_RADIENT_API_BASE_URL  # "https://api.radienthq.com/v1", a sign-in's stamp
SESSION = "sess_cos"
TENANT = "tnt_1"
ENDPOINT = "bep_1"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for name in ("RADIENT_API_KEY", "RADIENT_API_BASE_URL", "RADIENT_ORG_ALLOW_NONCANONICAL_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(AuthStore, "_org_hub_boundary", lambda self, boundary: None)
    return tmp_path


def _auth(root: Path) -> AuthStore:
    """The bridge's own credential root — the spelling the receiver opens."""
    return AuthStore(db_path=root / "auth.db", config_dir=root)


def _row(root: Path, *, account: str = ACCOUNT, hub: str = HUB, tag: str = "a") -> int:
    with closing(_auth(root)) as store:
        return store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": account,
                "api_endpoint": hub,
                "access": f"access-{tag}-test-only",
                "refresh": f"refresh-{tag}-test-only",
                # Far future (ms): the access is served as-is, never refreshed.
                "expires": 4_102_444_800_000,
            },
        ).id


def _gone_row(root: Path) -> int:
    """A row id this device used to hold — the incident's signed-out row 50."""
    with closing(_auth(root)) as store:
        row = store.upsert_credential(
            "radient",
            {"type": "oauth", "account_id": ACCOUNT, "api_endpoint": HUB, "access": "x"},
        )
        store._conn.execute("DELETE FROM auth_credentials WHERE id = ?", (row.id,))
        store._conn.commit()
    return row.id


def _duplicate(root: Path, credential_id: int) -> None:
    """A second same-account row (the store's dedupe normally prevents it)."""
    with closing(_auth(root)) as store:
        store._conn.execute(
            "INSERT INTO auth_credentials (provider, credential_type, data, identity_key,"
            " created_at, updated_at) SELECT provider, credential_type, data, NULL,"
            " created_at, updated_at FROM auth_credentials WHERE id = ?",
            (credential_id,),
        )
        store._conn.commit()


def _enroll(root: Path, pin: int) -> OwnerSnapshot:
    """The committed enrollment exactly as the operator's: enabled, designated."""
    owner = OwnerSnapshot(
        credential_id=pin,
        account_id=ACCOUNT,
        tenant_id=TENANT,
        endpoint_id=ENDPOINT,
        canonical_base_url=HUB,
        enrollment_generation=3,
    ).validated()
    Store(root).commit_enrollment(
        owner, OwnerReceipt(CONSENT_VERSION_V1, ENDPOINT), cos_session_id=SESSION
    )
    manager = ConfigManager(root)
    settings_io.write_setting(manager, settings_io.BY_KEY["team_bridge.enabled"], True)
    settings_io.write_setting(manager, settings_io.BY_KEY["team_bridge.session"], SESSION)
    return owner


def _enrollment(root: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(state_db_path(root))) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM enrollments").fetchone())


def _heals(root: Path) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(state_db_path(root))) as conn:
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM credential_heals")]
        except sqlite3.OperationalError:
            return []  # never created: no heal ever ran


def _hub_transport() -> httpx.MockTransport:
    """The hub, answering ``status`` and ``conversations`` for the enrolled endpoint."""

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Bearer access-")
        if request.url.path.endswith("/messaging/status"):
            result: dict[str, Any] = {
                "api_version": "1",
                "available": True,
                "consent": {"version": CONSENT_VERSION_V1, "text": "served consent"},
                "tenants": [
                    {
                        "tenant_id": TENANT,
                        "tenant_name": "Org",
                        "eligible": True,
                        "endpoint": {"id": ENDPOINT, "label": "My agent", "state": "active"},
                        "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
                    }
                ],
            }
        else:
            result = {
                "conversations": [{"id": "bcv_room0001", "kind": "channel", "member_count": 2}],
                "next_cursor": None,
            }
        return httpx.Response(200, json={"msg": "ok", "result": result})

    return httpx.MockTransport(respond)


async def _tool_answer(root: Path) -> dict[str, Any]:
    """What the model sees: a REAL session runtime, admitted, then the ``channels`` op."""
    runtime = host.bridge_runtime_for_session(
        config_manager=ConfigManager(root), session_id=SESSION
    )
    assert runtime is not None
    # The one injected seam: the transport (the client's own test contract).
    runtime.client._transport = _hub_transport()
    await host.admit_bridge_runtime(runtime)
    return await execute_bridge(runtime, {"op": "channels", "limit": 10})


async def _run_receiver(root: Path, owner: OwnerSnapshot, *, horizon: float) -> Any:
    """``__main__``'s loop body for one owner: a client pinned to the committed row,
    its real ``current_access`` as the access provider, the default owner source."""
    stop = asyncio.Event()
    with closing(_auth(root)) as auth_store:
        client = TeamBridgeClient(
            root, base_url=HUB, credential_id=owner.credential_id, store=auth_store
        )
        clock = _SimulatedClock(stop, horizon=horizon)
        real_pause = loop_mod._pause
        loop_mod._pause = clock.pause  # type: ignore[assignment]
        try:
            return await asyncio.wait_for(
                loop_mod.run_forever(
                    owner,
                    config_dir=root,
                    auth_store=auth_store,
                    access_provider=client.current_access,
                    stop=stop,
                ),
                timeout=20,
            )
        finally:
            loop_mod._pause = real_pause  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. The already-broken install heals on receiver start — no sign-in at all.
# ---------------------------------------------------------------------------


def test_a_broken_install_heals_on_receiver_start_without_a_login(root: Path) -> None:
    gone = _gone_row(root)
    fresh = _row(root)
    assert fresh != gone, "the re-login landed on a NEW row, as in the incident"
    owner = _enroll(root, gone)
    before = _enrollment(root)

    # The reproduction: today's answer to a signed-in operator.
    refused = asyncio.run(_tool_answer(root))
    assert refused["code"] == "bridge_credential_pin_gone", refused

    # The receiver starts (the updated build): one pass of the real loop.
    rearmed = asyncio.run(_run_receiver(root, owner, horizon=600))
    assert rearmed is not None, "the loop handed back a re-pointed owner to re-arm on"
    assert rearmed.credential_id == fresh
    assert rearmed.enrollment_generation == owner.enrollment_generation + 1

    after = _enrollment(root)
    assert after == {
        **before,
        "credential_id": fresh,
        "enrollment_generation": before["enrollment_generation"] + 1,
    }, "only the pin and its generation moved: consent and designation byte-identical"
    assert after["designated_session"] == SESSION
    heals = _heals(root)
    assert len(heals) == 1
    assert heals[0]["old_credential_id"] == gone and heals[0]["new_credential_id"] == fresh
    assert heals[0]["trigger"] == "receiver_start"
    assert heals[0]["account_id"] == ACCOUNT
    assert heals[0]["owner_key"] == before["owner_key"]
    assert heals[0]["enrollment_generation"] == after["enrollment_generation"]

    # An actual bridge call now succeeds: status over the transport, as the
    # re-armed receiver's client would make it ...
    client = TeamBridgeClient(
        root, base_url=HUB, credential_id=rearmed.credential_id, transport=_hub_transport()
    )
    status = asyncio.run(client.status())
    assert status["available"] is True
    # ... and the tool op returns channels instead of the refusal.
    answer = asyncio.run(_tool_answer(root))
    assert answer["ok"] is True and answer["state"] == "ok", answer
    listed = json.loads(answer["content"])
    assert [c["id"] for c in listed["conversations"]] == ["bcv_room0001"], answer


# ---------------------------------------------------------------------------
# 2. A verified login heals the bridge AND the tunnel, in the same sign-in.
# ---------------------------------------------------------------------------


def test_one_same_account_login_heals_both_the_bridge_and_the_tunnel(
    sandbox, monkeypatch  # noqa: F811 — the tunnel tests' fixture, imported above
) -> None:
    rootdir: Path = sandbox["root"]
    gone = _seed_gone_pin(account=ACCOUNT)  # the ambient store IS rootdir/auth.db
    _configure(gone, record={"id": "tunnel-1", "status": "active", "owner_account_id": ACCOUNT})
    _enroll(rootdir, gone)
    before = _enrollment(rootdir)

    with closing(AuthStore()) as store:
        assert store.db_path.parent == rootdir
        _stub_the_identity_provider(monkeypatch, account=ACCOUNT)
        assert auth_cli.run_login("radient", store.db_path.parent, store) == 0
        fresh = store.list_credentials("radient")[0].id
    assert fresh != gone

    assert tunnel_config.load()["credential_id"] == fresh, "the tunnel heal still works"
    after = _enrollment(rootdir)
    assert after["credential_id"] == fresh
    assert after["enrollment_generation"] == before["enrollment_generation"] + 1
    assert after["designated_session"] == SESSION
    assert (after["consent_version"], after["consent_ref"]) == (
        before["consent_version"],
        before["consent_ref"],
    )
    heals = _heals(rootdir)
    assert [(h["old_credential_id"], h["new_credential_id"], h["trigger"]) for h in heals] == [
        (gone, fresh, "login")
    ]


# ---------------------------------------------------------------------------
# 3. A different identity, or two candidates: nothing written, honest refusal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["different_account", "two_candidates"])
def test_an_unprovable_successor_is_never_re_pointed(root: Path, shape: str) -> None:
    gone = _gone_row(root)
    if shape == "different_account":
        _row(root, account="someone-else")
    else:
        _duplicate(root, _row(root))
    owner = _enroll(root, gone)
    database = state_db_path(root)
    snapshot = (_enrollment(root), loop_mod._owner_signal(root), database.read_bytes())

    rearmed = asyncio.run(_run_receiver(root, owner, horizon=600))
    assert rearmed is None, "no owner change: the loop idled until stop"

    assert (_enrollment(root), loop_mod._owner_signal(root), database.read_bytes()) == snapshot
    assert _heals(root) == []
    refused = asyncio.run(_tool_answer(root))
    assert refused["code"] == "bridge_credential_pin_gone", refused
    assert "no longer on this device" in refused["error"]
    assert "sign in to Radient to use agent messages" not in refused["error"]


def test_a_genuinely_signed_out_device_keeps_the_sign_in_sentence(root: Path) -> None:
    _enroll(root, _gone_row(root))  # the pin is gone and NO login remains
    refused = asyncio.run(_tool_answer(root))
    assert (refused.get("code"), refused.get("error")) == (
        "bridge_signin_required",
        "sign in to Radient to use agent messages",
    ), refused


# ---------------------------------------------------------------------------
# 4. Idempotence: a healthy pin is never churned by the start path.
# ---------------------------------------------------------------------------


def test_a_live_pin_is_never_churned_across_repeated_refusals(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm runs on every sign-in refusal; with the pin PRESENT and live (a
    transient refresh refusal, say) it must write nothing, however often."""
    pin = _row(root)
    _row(root, tag="b", hub="https://other-hub.example/v1")  # unrelated noise
    owner = _enroll(root, pin)
    database = state_db_path(root)
    snapshot = (_enrollment(root), loop_mod._owner_signal(root), database.read_bytes())
    attempts: list[int] = []

    async def refuse() -> Any:
        attempts.append(1)
        raise BridgeSignInRequired("refresh refused")

    async def drive() -> Any:
        stop = asyncio.Event()
        clock = _SimulatedClock(stop, horizon=loop_mod.OWNER_RECHECK_S * 10)
        monkeypatch.setattr(loop_mod, "_pause", clock.pause)
        return await asyncio.wait_for(
            loop_mod.run_forever(
                owner, config_dir=root, auth_store=None, access_provider=refuse, stop=stop
            ),
            timeout=20,
        )

    assert asyncio.run(drive()) is None
    assert len(attempts) >= 9, "the refusal arm really ran repeatedly"
    assert (_enrollment(root), loop_mod._owner_signal(root), database.read_bytes()) == snapshot
    assert _heals(root) == []

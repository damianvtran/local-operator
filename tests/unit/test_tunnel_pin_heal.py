"""A verified same-account sign-in re-points a tunnel pinned to a dead login.

The 2026-11-06 incident, reproduced from its real state: `tunnel/config.json`
pinned an OLD Radient row (legacy identity key, `account_id` but no
`api_endpoint`, `grant_dead_at` set), and every sign-in — desktop, TUI, `lop
login radient` — wrote a NEW row for the same account beside it. The verdict
read the pinned row forever (`login_required`), its remedy was another sign-in,
and the re-arm declined because the new row was not the pinned one.

Driven through the REAL completion path (`AuthStore.complete_radient_login`
with a verified flow receipt; only HTTP is synthetic) and the real tunnel
config file; only the supervisor seam is stubbed, so a broken guard shows up
as a recorded `launchctl` call rather than as a restart of a live connector.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from local_operator.providers import auth_cli
from local_operator.providers.auth_store import GRANT_DEAD_AT_KEY, AuthStore
from local_operator.providers.registry import get_provider_definition
from local_operator.tunnels import api, config, gateway, install, report, state
from local_operator.tunnels.arguments import add_parser
from local_operator.tunnels.cli import dispatch
from tests.unit.providers.test_radient_account_retention import _verified_login_result
from tests.unit.providers.test_radient_login_admission import _quiet_host
from tests.unit.test_tunnels import (  # noqa: F401 — shared builder helpers
    _Connector,
    _synthetic_port,
    make_connection,
    make_signing_key,
)

ACCOUNT = "acct-operator"


# The two fixtures below are LOCAL wrappers over `test_tunnels`' plain builders,
# and the indirection is the point: importing a fixture object and then naming it
# as a parameter here is an F811 redefinition, so the shared body lives in a
# function and each module wraps its own fixture around it (the same rule
# `tests/unit/aida/conftest.py` documents for its root helper).
@pytest.fixture(scope="module")
def signing_key():
    return make_signing_key()


@pytest.fixture
def connection(signing_key):
    return make_connection(signing_key)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """An isolated store root with the supervisor seam recording, not acting."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_TUNNEL_NO_REARM", raising=False)
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    # The org Agent Hub refresh a sign-in starts is a network call with these
    # synthetic tokens; it is not what this file is about.
    monkeypatch.setattr(AuthStore, "_org_hub_boundary", lambda self, boundary: None)
    plist = tmp_path / "com.local-operator.tunnel.plist"
    plist.write_text("plist")
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: install.supervisors.LAUNCHCTL)
    monkeypatch.setattr(install, "service_path", lambda: plist)
    monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda _base: True)
    kicks: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        install,
        "_launchctl",
        lambda *args: kicks.append(args) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    return {"kicks": kicks, "plist": plist, "root": tmp_path}


def _seed_dead_legacy_pin(*, account: str = ACCOUNT, endpoint: str | None = None) -> int:
    """The pinned row exactly as the incident found it."""
    data: dict[str, Any] = {
        "type": "oauth",
        "account_id": account,
        "access": "dead-access-test-only",
        "refresh": "dead-refresh-test-only",
        "expires": 1,
    }
    if endpoint is not None:
        data["api_endpoint"] = endpoint
    with closing(AuthStore()) as store:
        row = store.upsert_credential("radient", data)
        # Written AFTER the upsert, straight into the row, as the store's own
        # refusal handler leaves it: a credential write supersedes the
        # tombstone, so passing it to `upsert_credential` would not persist it.
        # And it must be there: a dead-grant row is answered without a refresh
        # POST, so no test here can reach the real token endpoint.
        dead = {**data, GRANT_DEAD_AT_KEY: int(time.time() * 1000) - 60_000}
        # The legacy identity key (pre endpoint-stamping): a plain account id,
        # which is why the fresh sign-in's upsert never matches this row.
        store._conn.execute(
            "UPDATE auth_credentials SET identity_key = ?, data = ? WHERE id = ?",
            (account, json.dumps(dead), row.id),
        )
        store._conn.commit()
    return row.id


def _configure(pin: int, **extra: Any) -> dict[str, Any]:
    value = {
        "tunnel_id": "tunnel-1",
        "credential_id": pin,
        "gateway_port": 4099,
        "record": {"id": "tunnel-1", "status": "active"},
        "cloudflared_path": "/opt/bin/cloudflared",
        "mobile_password": "kept-test-only",
        **extra,
    }
    config.save(value)
    return value


def _sign_in(account: str = ACCOUNT) -> int:
    """A completed verified Radient login, through the store's real path."""
    result = asyncio.run(_verified_login_result(account))
    with closing(AuthStore()) as store:
        return store.complete_radient_login(result).id


def test_incident_state_is_healed_by_a_same_account_sign_in(sandbox) -> None:
    pin = _seed_dead_legacy_pin()
    before = _configure(pin)
    state.mark_parked(reason=gateway.LOGIN_REQUIRED, detail="x", credential_id=pin)

    # The incident: the pinned row is dead, and that is all the verdict can see.
    assert asyncio.run(report.login_verdict(config.load())) == {
        "credential_id": pin,
        "state": gateway.LOGIN_REQUIRED,
    }

    fresh = _sign_in()

    assert fresh != pin, "the sign-in must land on a NEW row, as in the incident"
    after = config.load()
    assert after == {**before, "credential_id": fresh}, "only the pin moves"
    assert sandbox["kicks"] == [("kickstart", "-k", f"gui/{install.os.getuid()}/{install.LABEL}")]
    assert asyncio.run(report.login_verdict(after)) == {"credential_id": fresh, "state": "ok"}


def test_a_running_but_refused_connector_is_restarted_too(sandbox) -> None:
    pin = _seed_dead_legacy_pin()
    _configure(pin)  # no park: the connector is up, being refused
    fresh = _sign_in()
    assert config.load()["credential_id"] == fresh
    assert len(sandbox["kicks"]) == 1 and sandbox["kicks"][0][:2] == ("kickstart", "-k")


def test_a_different_account_never_moves_the_pin(sandbox) -> None:
    pin = _seed_dead_legacy_pin()
    _configure(pin)
    _sign_in("someone-else")
    assert config.load()["credential_id"] == pin
    assert sandbox["kicks"] == []


def test_a_cached_owner_that_disagrees_with_the_pinned_row_blocks_the_heal(sandbox) -> None:
    pin = _seed_dead_legacy_pin()
    _configure(pin, record={"id": "tunnel-1", "owner_account_id": "other-owner"})
    _sign_in()
    assert config.load()["credential_id"] == pin


def test_a_stopped_tunnel_is_left_alone(sandbox) -> None:
    pin = _seed_dead_legacy_pin()
    _configure(pin, stopped=True)
    _sign_in()
    assert config.load()["credential_id"] == pin
    assert sandbox["kicks"] == []


def test_a_pin_bound_to_another_hub_is_left_alone(sandbox) -> None:
    pin = _seed_dead_legacy_pin(endpoint="https://hub.example/v1")
    _configure(pin)
    _sign_in()
    assert config.load()["credential_id"] == pin
    assert sandbox["kicks"] == []


def test_a_live_pin_is_the_operators_choice(sandbox) -> None:
    with closing(AuthStore()) as store:
        live = store.upsert_credential(
            "radient",
            {"type": "oauth", "account_id": ACCOUNT, "access": "a", "refresh": "r"},
        )
    _configure(live.id)
    _sign_in()
    assert config.load()["credential_id"] == live.id


def test_no_rearm_re_points_the_file_but_never_starts_a_service(sandbox, monkeypatch) -> None:
    """The opt-out gates the SERVICE, as its docstring says; the file is the store's own."""
    monkeypatch.setenv("LOP_TUNNEL_NO_REARM", "1")
    pin = _seed_dead_legacy_pin()
    _configure(pin)
    fresh = _sign_in()
    assert config.load()["credential_id"] == fresh
    assert sandbox["kicks"] == []


def test_a_sandboxed_store_never_reaches_the_real_supervisor(sandbox, monkeypatch) -> None:
    monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda _base: False)
    pin = _seed_dead_legacy_pin()
    _configure(pin)
    fresh = _sign_in()
    assert config.load()["credential_id"] == fresh
    assert sandbox["kicks"] == []


def test_a_generic_credential_write_proves_nothing(sandbox) -> None:
    """Only the VERIFIED completion may re-point: an upsert carries no identity proof."""
    pin = _seed_dead_legacy_pin()
    _configure(pin)
    with closing(AuthStore()) as store:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": ACCOUNT,
                "api_endpoint": api.TUNNEL_HUB,
                "access": "a",
                "refresh": "r",
            },
        )
    assert config.load()["credential_id"] == pin
    assert sandbox["kicks"] == []


def test_a_gone_pin_heals_only_with_the_cached_owner_as_evidence(sandbox) -> None:
    _configure(4242)
    _sign_in()
    assert config.load()["credential_id"] == 4242, "no evidence of the owner: untouched"

    _configure(4242, record={"id": "tunnel-1", "owner_account_id": ACCOUNT})
    fresh = _sign_in()
    assert config.load()["credential_id"] == fresh


def test_cli_fallback_and_remedy_name_the_re_point_when_the_heal_did_not_run(sandbox) -> None:
    """A successor already here (written by a build without the heal, or while
    stopped): bare `lop tunnel configure` must pick it, and no surface may offer
    the sign-in that already happened."""
    pin = _seed_dead_legacy_pin()
    value = _configure(pin)
    with closing(AuthStore()) as store:
        fresh = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": ACCOUNT,
                "api_endpoint": api.TUNNEL_HUB,
                "access": "a",
                "refresh": "r",
                "expires": int(time.time() * 1000) + 3_600_000,
            },
        ).id

    assert api.usable_credential_id(pin) == fresh
    assert api.credential_id(api.usable_credential_id(pin)) == fresh

    login = asyncio.run(report.login_verdict(value))
    assert login == {
        "credential_id": pin,
        "state": gateway.LOGIN_REQUIRED,
        "successor_credential_id": fresh,
    }
    parked = {
        "state": "parked",
        "reason": gateway.LOGIN_REQUIRED,
        "remedy": {"command": "lop login radient", "url": gateway.CONSOLE_URL},
    }
    expected = {
        "kind": "run",
        "command": gateway.TERMINAL_REMEDY[gateway.LOGIN_SUPERSEDED],
        "url": gateway.CONSOLE_URL,
    }
    assert expected["command"] == "lop tunnel configure"
    assert report.remedy(value, parked, login) == expected
    assert report.remedy(value, {"state": "not serving", "reason": ""}, login) == expected


def test_two_same_account_successors_are_not_a_choice_made_here(sandbox) -> None:
    """Ambiguity declines; the store's dedupe normally prevents it, so it is
    planted by copying the row, the shape a hand-repaired database could hold."""
    pin = _seed_dead_legacy_pin()
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": ACCOUNT,
                "api_endpoint": api.TUNNEL_HUB,
                "access": "a",
                "refresh": "r",
            },
        )
        store._conn.execute(
            "INSERT INTO auth_credentials (provider, credential_type, data, identity_key,"
            " created_at, updated_at) SELECT provider, credential_type, data, NULL,"
            " created_at, updated_at FROM auth_credentials WHERE id = ?",
            (row.id,),
        )
        store._conn.commit()
    assert api.same_account_successor(pin) is None
    assert api.usable_credential_id(pin) == pin


def test_a_successor_on_another_hub_is_not_this_tunnels_login(sandbox) -> None:
    pin = _seed_dead_legacy_pin()
    with closing(AuthStore()) as store:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": ACCOUNT,
                "api_endpoint": "https://hub.example/v1",
                "access": "a",
                "refresh": "r",
            },
        )
    assert api.same_account_successor(pin) is None


def test_a_fresh_login_does_not_win_over_another_live_same_account_row(sandbox) -> None:
    """Agent review round 1, m-2: `fresh_id` is membership, not a filter.

    A live same-account row the operator already had makes the store
    ambiguous for status and configure; the heal must call it the same.
    """
    pin = _seed_dead_legacy_pin()
    _configure(pin)
    with closing(AuthStore()) as store:
        # Unkeyed and stamped with the hub's trailing-slash spelling: the same
        # hub to the tunnel (`api._hub` folds it), a different binding to the
        # store, so the sign-in below cannot dedupe onto it.
        store._conn.execute(
            "INSERT INTO auth_credentials (provider, credential_type, data, identity_key,"
            " created_at, updated_at) VALUES ('radient', 'oauth', ?, NULL, 1, 1)",
            (
                json.dumps(
                    {
                        "type": "oauth",
                        "account_id": ACCOUNT,
                        "api_endpoint": api.TUNNEL_HUB + "/",
                        "access": "a",
                        "refresh": "r",
                    }
                ),
            ),
        )
        store._conn.commit()
    fresh = _sign_in()
    assert api.same_account_successor(pin, fresh_id=fresh) is None
    assert config.load()["credential_id"] == pin
    assert sandbox["kicks"] == []


def test_a_pin_with_no_account_heals_on_the_cached_owner(sandbox) -> None:
    """Agent review round 1, m-4: a pre-identity pin row has no `account_id`;
    the cached owner is the tunnel's owner itself, so it is the evidence."""
    with closing(AuthStore()) as store:
        pin = store.upsert_credential(
            "radient", {"type": "oauth", "access": "old", "refresh": "old", "expires": 1}
        ).id
    _configure(pin)
    _sign_in()
    assert config.load()["credential_id"] == pin, "no account anywhere: untouched"

    _configure(pin, record={"id": "tunnel-1", "owner_account_id": ACCOUNT})
    fresh = _sign_in()
    assert config.load()["credential_id"] == fresh


def test_a_login_into_another_store_never_re_points_the_default_tunnel(sandbox) -> None:
    """Agent review round 1, m-1: the heal's proof reads the default store, so
    an id written by a store on another `db_path` must not be taken for a row
    there — even when that id names a provable successor in the default one."""
    pin = _seed_dead_legacy_pin()
    _configure(pin)
    with closing(AuthStore()) as store:
        successor = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": ACCOUNT,
                "api_endpoint": api.TUNNEL_HUB,
                "access": "a",
                "refresh": "r",
            },
        ).id
    other_root = sandbox["root"] / "other"
    other_root.mkdir()
    result = asyncio.run(_verified_login_result(ACCOUNT))
    with closing(AuthStore(db_path=other_root / "auth.db", config_dir=other_root)) as other:
        while True:  # line the other store's next id up with the default successor
            nxt = other._conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM auth_credentials"
            ).fetchone()[0]
            if nxt >= successor:
                break
            other._conn.execute(
                "INSERT INTO auth_credentials (provider, credential_type, data, created_at,"
                " updated_at) VALUES ('filler', 'api_key', '{}', 1, 1)"
            )
            other._conn.commit()
        written = other.complete_radient_login(result).id
    assert written == successor
    assert config.load()["credential_id"] == pin
    assert sandbox["kicks"] == []


# ---------------------------------------------------------------------------
# The operator's acceptance, on the REAL command paths.
#
# The tests above drive the store's completion hook and the verdict functions;
# what the 2026-11-06 incident was reported as is a pair of COMMANDS —
# `lop login radient` did not bring the tunnel back, and `lop tunnel status`
# said it would. These two run those commands (`auth_cli.run_login` and the
# `tunnel status --json` parser entry point), with only the identity provider
# synthetic, and read the JSON the second one really prints.
# ---------------------------------------------------------------------------


def _seed_gone_pin(*, account: str = ACCOUNT) -> int:
    """A config pinned to a credential row this device no longer has.

    That is the incident's own state, not a stand-in for it: signing out
    DELETED row 135 while `tunnel/config.json` kept naming it, so the verdict
    read `owner_missing` and the only local evidence of the owner left was the
    cached `record.owner_account_id`.
    """
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": account,
                "access": "removed-with-the-sign-out",
                "refresh": "removed-with-the-sign-out",
            },
        )
        store._conn.execute("DELETE FROM auth_credentials WHERE id = ?", (row.id,))
        store._conn.commit()
    return row.id


def _incident_tunnel(connection: dict[str, Any], pin: int, *, harness_port: int) -> dict[str, Any]:
    """The configuration the incident had: pinned to the gone row, owner cached.

    Ports are synthetic on purpose — this file never dials a harness, and the
    approved harness port in the stored record has to match the one the stubbed
    relay serves for the connector to publish at all.
    """
    gateway_port = _synthetic_port()
    while gateway_port == harness_port:  # two distinct loopback ports
        gateway_port = _synthetic_port()
    record = copy.deepcopy(connection["tunnel"])
    record["gateway_port"] = gateway_port
    record["owner_account_id"] = ACCOUNT
    record["harnesses"][0]["port"] = harness_port
    return {
        "tunnel_id": connection["tunnel"]["id"],
        "credential_id": pin,
        "gateway_port": gateway_port,
        "record": record,
        "cloudflared_path": "/opt/bin/cloudflared",
        "mobile_password": "kept-test-only",
    }


def _stub_the_identity_provider(monkeypatch: pytest.MonkeyPatch, *, account: str) -> None:
    """`lop login radient`'s real path, with only the IdP's HTTP synthetic.

    Same seam the admission tests use: the provider definition's `login`
    callable is replaced, and everything after it — the verified-receipt check,
    `AuthStore.complete_radient_login`, the credential-write hook and the heal
    it reaches — is production code.
    """
    _quiet_host(monkeypatch)
    definition = get_provider_definition("radient")
    assert definition is not None

    async def login(callbacks: Any, *, signal: Any, **kwargs: Any) -> dict[str, Any]:
        return await _verified_login_result(account, signal=signal)

    monkeypatch.setattr(
        auth_cli, "get_provider_definition", lambda _: replace(definition, login=login)
    )


def _status_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    return parser


async def _serve_the_connector_and_read_status(
    connection: dict[str, Any],
    stored: dict[str, Any],
    parser: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Start the real connector and read `lop tunnel status --json` while it serves.

    `service.run()` is the code the heal's `kickstart` launches, so this is the
    production startup path — it reads the re-pointed configuration, connects
    through the stubbed relay, withdraws the park and serves `/_lop_tunnel/health`
    on a real loopback listener. The health probe in `report.probe` is therefore a
    real request, not a fixture of one, which is the only way `connector.state`
    can honestly read `connected`.
    """
    from local_operator.tunnels import cli, service

    served = copy.deepcopy(connection)
    served["gateway_port"] = served["tunnel"]["gateway_port"] = stored["gateway_port"]
    served["tunnel"]["harnesses"][0]["port"] = stored["record"]["harnesses"][0]["port"]

    async def request(method: str, path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """The relay, as the connector uses it: /connect, then a record per poll.

        The poller compares the record it reads against the connection it made —
        a version or status it does not recognise makes the unit RESTART rather
        than keep serving — so the stub has to answer both, or the teardown below
        would be measuring a restart instead of the graceful stop it asserts.
        """
        return served if path.endswith("/connect") else served["tunnel"]

    relay = AsyncMock()
    relay.request.side_effect = request
    monkeypatch.setattr(service, "RadientTunnels", lambda *_: relay)
    monkeypatch.setattr(service, "cloudflared_binary", lambda *_: "/trusted/cloudflared")
    monkeypatch.setattr(service, "load_password", lambda: "private-local-password")

    async def start_connector(*args: Any, **kwargs: Any) -> _Connector:
        """cloudflared, as far as this process can tell — it writes its pidfile.

        `Gateway.connector_ready` is exactly "the ready file exists and the child
        is alive", and a real cloudflared creates that file (`--pidfile`). A stub
        that skipped it would leave a serving gateway answering `connected:
        false`, which is not the state this test exists to prove.
        """
        (config.directory() / "cloudflared.pid").write_text(str(os.getpid()))
        return _Connector()

    monkeypatch.setattr(
        service.asyncio, "create_subprocess_exec", AsyncMock(side_effect=start_connector)
    )
    task = asyncio.create_task(service.run())
    try:
        health: dict[str, Any] = {}
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(200):
                await asyncio.sleep(0.05)
                try:
                    reply = await client.get(
                        f"http://127.0.0.1:{stored['gateway_port']}/_lop_tunnel/health"
                    )
                except httpx.ConnectError:
                    # Not bound yet: `run()` reconciles its pin (a store read,
                    # off-thread) before it binds, so the first probes can
                    # land before the listener exists.
                    continue
                health = reply.json()
                if health.get("ok") is True and health.get("connected") is True:
                    break
        assert health.get("ok") is True and health.get("connected") is True, health
        assert state.read() is None, "a serving connector withdraws the park itself"

        # The acceptance, read from the operator's own surface.
        live = AsyncMock()
        live.request.return_value = connection["tunnel"]
        monkeypatch.setattr(cli, "RadientTunnels", lambda *_: live)
        return json.loads(await dispatch(parser.parse_args(["tunnel", "status", "--json"])))
    finally:
        stopping = config.load()
        stopping["stopped"] = True
        config.save(stopping)
        assert await asyncio.wait_for(task, timeout=15) == 0


def test_the_real_login_command_brings_the_tunnel_back_with_no_connect_call(
    sandbox, connection, monkeypatch
) -> None:
    """Acceptance (2026-11-06): `lop login radient` alone leaves the phone link up.

    Sign-out → sign-in as the SAME identity, through the commands the operator
    actually typed, and the JSON `lop tunnel status` really prints:
    `connector.state` ends `connected` and `login.state` ends `ok` with no
    `lop tunnel connect` anywhere in the test. The connector is started by the
    real `service.run()` — the same code the re-point's `kickstart` launches —
    so the park is withdrawn by the connector itself and the health probe is a
    real loopback request, not a stub.
    """
    from local_operator.tunnels import cli

    gone = _seed_gone_pin()
    stored = _incident_tunnel(connection, gone, harness_port=_synthetic_port())
    config.save(stored)
    state.mark_parked(
        reason=gateway.LOGIN_REQUIRED,
        detail=gateway.TERMINAL_DETAIL[gateway.LOGIN_REQUIRED],
        credential_id=gone,
    )
    offline = AsyncMock()
    offline.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: offline)
    parser = _status_parser()

    # The incident, as `lop tunnel status --json` reported it.
    before = json.loads(asyncio.run(dispatch(parser.parse_args(["tunnel", "status", "--json"]))))
    assert before["connector"]["state"] == "parked"
    assert before["connector"]["reason"] == gateway.LOGIN_REQUIRED
    assert before["login"]["credential_id"] == gone
    assert before["login"]["state"] == gateway.OWNER_MISSING
    # That read is the OLD self-heal (it kicks a parked connector), and on this
    # state it is the loop the operator was in: the kick starts a connector whose
    # pin names a row that no longer exists. Cleared so the count below is the
    # login's alone.
    sandbox["kicks"].clear()

    # ... and the command that was supposed to fix it.
    with closing(AuthStore()) as store:
        _stub_the_identity_provider(monkeypatch, account=ACCOUNT)
        assert auth_cli.run_login("radient", store.db_path.parent, store) == 0
        fresh = store.list_credentials("radient")[0].id
    assert fresh != gone, "the sign-in lands on a NEW row, as in the incident"

    assert config.load()["credential_id"] == fresh, "the sign-in re-pointed the pin by itself"
    assert config.load() == {
        **stored,
        "credential_id": fresh,
    }, "only the pin moved: no connect ran, and the cloud record is untouched"
    expected_kick = ("kickstart", "-k", f"gui/{install.os.getuid()}/{install.LABEL}")
    assert sandbox["kicks"] == [
        expected_kick
    ], "the heal restarted the connector once, and nothing else did"
    receipt = install.last_heal_receipt()
    assert receipt is not None, "the re-bind is recorded where it can be audited"
    assert receipt["old_credential_id"] == gone and receipt["new_credential_id"] == fresh
    assert receipt["tunnel_id"] == connection["tunnel"]["id"]
    assert receipt["account_id"] == ACCOUNT
    assert receipt["trigger"] == install.HEAL_TRIGGER_LOGIN == "login"
    assert isinstance(receipt["at"], int) and receipt["at"] > 0

    # The connector the restart launches, run for real: it reads the re-pointed
    # configuration, connects through the (stubbed) relay and serves its gateway.
    after = asyncio.run(
        _serve_the_connector_and_read_status(connection, stored, parser, monkeypatch)
    )
    assert after["connector"]["state"] == "connected", after["connector"]
    assert after["login"]["state"] == "ok", after["login"]
    assert after["login"]["credential_id"] == fresh
    assert after["remedy"] is None, "nothing left to run"


def test_a_different_account_signed_in_leaves_the_tunnel_alone_and_says_so(
    sandbox, connection, monkeypatch
) -> None:
    """Acceptance, negative half: another identity never repoints a tunnel.

    The heal's whole safety is the identity check, so the case it refuses is
    pinned through the same commands: after a sign-in as SOMEONE ELSE the pin
    still names the row the tunnel was enrolled with, nothing was restarted, no
    re-bind receipt exists — and the surface offers the remedy that can work
    (the re-point) while promising no sign-in that cannot.
    """
    gone = _seed_gone_pin()
    stored = _incident_tunnel(connection, gone, harness_port=_synthetic_port())
    config.save(stored)
    state.mark_parked(
        reason=gateway.LOGIN_REQUIRED,
        detail=gateway.TERMINAL_DETAIL[gateway.LOGIN_REQUIRED],
        credential_id=gone,
    )
    from local_operator.tunnels import cli

    offline = AsyncMock()
    offline.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: offline)
    parser = _status_parser()
    asyncio.run(dispatch(parser.parse_args(["tunnel", "status", "--json"])))
    sandbox["kicks"].clear()

    with closing(AuthStore()) as store:
        _stub_the_identity_provider(monkeypatch, account="someone-else")
        assert auth_cli.run_login("radient", store.db_path.parent, store) == 0

    assert config.load() == stored, "a different identity moves neither the pin nor anything else"
    assert sandbox["kicks"] == [], "and never restarts a connector at the wrong login"
    # The receipt's documented name, spelled here rather than imported: this
    # assertion has to mean the same thing on a build that has no heal at all.
    assert not (config.directory() / "login-heal.json").exists()

    payload = json.loads(asyncio.run(dispatch(parser.parse_args(["tunnel", "status", "--json"]))))
    assert payload["connector"]["state"] == "parked"
    assert payload["login"]["credential_id"] == gone
    assert payload["login"]["state"] == gateway.OWNER_MISSING
    assert payload["remedy"]["command"] == gateway.TERMINAL_REMEDY[gateway.OWNER_MISSING]
    assert payload["remedy"]["command"] == "lop tunnel configure"
    # The 2026-11-06 defect in one assertion: the shared sentence used to say a
    # sign-in starts the connector again, unconditionally, on a tunnel where it
    # could not. It now names the one sign-in that does.
    detail = payload["connector"]["detail"].lower()
    assert "only signing in to that same account restarts it" in detail


@pytest.mark.parametrize(
    ("cause", "detail"),
    [
        ("no_unit", "HEAL_UNIT_MISSING"),
        ("no_supervisor", "HEAL_NO_SERVICE_MANAGER"),
        ("refused", "HEAL_RESTART_REFUSED"),
        ("other_store", "HEAL_NOT_THIS_STORE"),
    ],
)
def test_a_heal_that_cannot_start_the_connector_says_which_reason(
    sandbox, connection, monkeypatch, cause: str, detail: str
) -> None:
    """Requirement 4: the park names the REAL cause instead of `login_required`.

    The heal can re-point the file and still not get a connector running — no
    service manager, no installed unit, a store outside the real home, or a
    supervisor that refuses. Every one of those used to leave the park saying
    `login_required`, i.e. the sign-in the operator had just completed, with a
    remedy (`lop login radient`) that cannot start a service. The reason code is
    `LOCAL_PREREQUISITE` — the existing word for this class — and the detail is
    one fixed literal per cause, so the failure is auditable in `state.json`
    without a path or a credential in it.
    """
    from local_operator.tunnels import cli

    if cause == "no_unit":
        monkeypatch.setattr(install, "service_path", lambda: sandbox["root"] / "absent.plist")
    elif cause == "no_supervisor":
        monkeypatch.setattr(install.supervisors, "supervisor", lambda: None)
    elif cause == "other_store":
        monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda _base: False)
    else:
        # Recorded as well as refused: the attempt is the evidence this cause
        # differs from the three that never reach the supervisor at all.
        monkeypatch.setattr(
            install,
            "_launchctl",
            lambda *args: (
                sandbox["kicks"].append(args)
                or SimpleNamespace(returncode=1, stdout="", stderr="kickstart: no such process")
            ),
        )

    gone = _seed_gone_pin()
    stored = _incident_tunnel(connection, gone, harness_port=_synthetic_port())
    config.save(stored)
    state.mark_parked(
        reason=gateway.LOGIN_REQUIRED,
        detail=gateway.TERMINAL_DETAIL[gateway.LOGIN_REQUIRED],
        credential_id=gone,
    )

    with closing(AuthStore()) as store:
        _stub_the_identity_provider(monkeypatch, account=ACCOUNT)
        assert auth_cli.run_login("radient", store.db_path.parent, store) == 0
        fresh = store.list_credentials("radient")[0].id
    assert config.load()["credential_id"] == fresh, "the re-point still happened"
    assert len(sandbox["kicks"]) == (1 if cause == "refused" else 0), sandbox["kicks"]
    receipt = install.last_heal_receipt()
    assert (
        receipt is not None and receipt["new_credential_id"] == fresh
    ), "the re-bind is auditable even when the restart is not"

    offline = AsyncMock()
    offline.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: offline)
    parser = _status_parser()
    payload = json.loads(asyncio.run(dispatch(parser.parse_args(["tunnel", "status", "--json"]))))

    assert payload["connector"]["reason"] == gateway.LOCAL_PREREQUISITE
    assert payload["connector"]["detail"] == getattr(install, detail)
    assert payload["connector"]["state"] == "parked"
    assert payload["login"]["state"] == "ok", "the login is fine; this is not a sign-in problem"
    assert payload["remedy"]["command"] == gateway.TERMINAL_REMEDY[gateway.LOCAL_PREREQUISITE]
    assert gateway.reason_label(payload["connector"]["reason"]) == "prerequisite missing"


# ---------------------------------------------------------------------------
# START-TIME RECONCILE (2026-11-06, v0.101.6). The operator signed out and back
# in as the same account while the machine still ran a build with no heal, so
# the pin went 136 -> 137 with nothing to re-point it; the update that shipped
# the login heal found the connector parked `login_required` / `owner_missing`
# with the OLD build's sentence in `state.json`, and stayed there, because the
# login heal only runs from a verified sign-in. These pin the connector's own
# start-time reconcile (`install.heal_pin_at_start`, called by `service.run`)
# and the update refresh that starts a parked connector for it.
# ---------------------------------------------------------------------------

# What the pre-#3519 build persisted, verbatim in shape: a sentence the current
# vocabulary no longer says, so a re-park on THIS build is visible as a change.
_OLD_BUILD_DETAIL = (
    "Your Radient login needs attention. The connector stopped and will not retry"
    " by itself. Signing in again starts it again on its own."
)


def _no_login_may_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The acceptance is "no sign-in": any login-path heal call fails the test."""

    def refuse(**_kwargs: Any) -> str:
        pytest.fail("the start-time heal must not depend on a login")

    monkeypatch.setattr(install, "heal_pin_after_login", refuse)


def _live_shape(connection: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    """Pin -> gone row; the sole same-account row was written while NO heal ran.

    The successor is signed in BEFORE the tunnel configuration exists, so the
    login hook finds nothing to re-point — the exact position an install that
    broke on an older build is in when the new build first starts.
    """
    gone = _seed_gone_pin()
    fresh = _sign_in()
    stored = _incident_tunnel(connection, gone, harness_port=_synthetic_port())
    config.save(stored)
    state.mark_parked(reason=gateway.LOGIN_REQUIRED, detail=_OLD_BUILD_DETAIL, credential_id=gone)
    return gone, fresh, stored


def test_the_connector_heals_a_replaced_pin_when_it_starts_with_no_sign_in(
    sandbox, connection, monkeypatch
) -> None:
    """Acceptance: the live v0.101.6 state comes back by the connector starting."""
    from local_operator.tunnels import cli

    gone, fresh, stored = _live_shape(connection)
    assert fresh != gone
    offline = AsyncMock()
    offline.request.side_effect = httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(cli, "RadientTunnels", lambda *_: offline)
    parser = _status_parser()
    before = json.loads(asyncio.run(dispatch(parser.parse_args(["tunnel", "status", "--json"]))))
    assert before["connector"]["state"] == "parked"
    assert before["connector"]["detail"] == _OLD_BUILD_DETAIL
    assert before["login"]["state"] == gateway.OWNER_MISSING
    _no_login_may_run(monkeypatch)

    after = asyncio.run(
        _serve_the_connector_and_read_status(connection, stored, parser, monkeypatch)
    )

    assert config.load() == {
        **stored,
        "credential_id": fresh,
        "stopped": True,
    }, "only the pin moved (`stopped` is the helper's own teardown)"
    receipt = install.last_heal_receipt()
    assert receipt is not None
    assert {k: receipt[k] for k in ("old_credential_id", "new_credential_id", "trigger")} == {
        "old_credential_id": gone,
        "new_credential_id": fresh,
        "trigger": "connector_start",
    }
    assert receipt["account_id"] == ACCOUNT
    assert receipt["tunnel_id"] == connection["tunnel"]["id"]
    assert after["connector"]["state"] == "connected", after["connector"]
    assert after["login"]["credential_id"] == fresh, after["login"]
    assert after["login"]["state"] == "ok", after["login"]
    assert after["remedy"] is None
    # The serving connector withdrew the old build's park (asserted inside the
    # helper too): no surface still carries the stale sentence.
    assert state.read() is None


def test_two_replacements_heal_to_the_newest_row(sandbox) -> None:
    """Any number of sign-outs while no heal ran: the sole live row wins."""
    gone = _seed_gone_pin()
    _seed_gone_pin()  # the intermediate row, itself signed out since
    fresh = _sign_in()
    _configure(gone, record={"id": "tunnel-1", "owner_account_id": ACCOUNT})

    assert install.heal_pin_at_start() == fresh
    assert config.load()["credential_id"] == fresh
    receipt = install.last_heal_receipt()
    assert receipt is not None and receipt["old_credential_id"] == gone
    assert receipt["trigger"] == install.HEAL_TRIGGER_CONNECTOR_START
    assert sandbox["kicks"] == [], "the connector heals in-process; nothing is kicked"


def test_a_live_pin_is_a_no_op_at_start(sandbox) -> None:
    fresh = _sign_in()
    before = _configure(fresh, record={"id": "tunnel-1", "owner_account_id": ACCOUNT})
    assert install.heal_pin_at_start() is None
    assert config.load() == before
    assert install.last_heal_receipt() is None


def _run_the_connector_to_its_park(monkeypatch: pytest.MonkeyPatch) -> int:
    """`service.main()` for real, with only the relay replaced by a login refusal.

    The refusal is the one the relay gives a pin it cannot authenticate, so the
    connector parks through its own classification — the detail it writes is
    THIS build's sentence, which is the stale-detail half of the acceptance.
    """
    from local_operator.tunnels import service

    relay = AsyncMock()
    relay.request.side_effect = service.LoginRequired("Radient login required.")
    monkeypatch.setattr(service, "RadientTunnels", lambda *_: relay)
    monkeypatch.setattr(service, "cloudflared_binary", lambda *_: "/trusted/cloudflared")
    return service.main()


@pytest.mark.parametrize("shape", ["different_account", "two_same_account_rows"])
def test_a_declined_start_heal_changes_nothing_and_re_parks_with_the_current_sentence(
    sandbox, connection, monkeypatch, shape: str
) -> None:
    gone = _seed_gone_pin()
    if shape == "different_account":
        _sign_in("someone-else")
    else:
        fresh = _sign_in()
        with closing(AuthStore()) as store:
            store._conn.execute(
                "INSERT INTO auth_credentials (provider, credential_type, data, identity_key,"
                " created_at, updated_at) SELECT provider, credential_type, data, NULL,"
                " created_at, updated_at FROM auth_credentials WHERE id = ?",
                (fresh,),
            )
            store._conn.commit()
    stored = _incident_tunnel(connection, gone, harness_port=_synthetic_port())
    config.save(stored)
    state.mark_parked(reason=gateway.LOGIN_REQUIRED, detail=_OLD_BUILD_DETAIL, credential_id=gone)
    _no_login_may_run(monkeypatch)

    assert _run_the_connector_to_its_park(monkeypatch) == 0

    assert config.load() == stored, "no re-point"
    assert install.last_heal_receipt() is None, "no receipt"
    parked = state.parked()
    assert parked is not None and parked["reason"] == gateway.LOGIN_REQUIRED
    assert parked["detail"] == gateway.TERMINAL_DETAIL[gateway.LOGIN_REQUIRED]
    assert parked["detail"] != _OLD_BUILD_DETAIL, "re-parked with the CURRENT sentence"

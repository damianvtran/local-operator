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
import json
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.providers.auth_store import GRANT_DEAD_AT_KEY, AuthStore
from local_operator.tunnels import api, config, gateway, install, report, state
from tests.unit.providers.test_radient_account_retention import _verified_login_result

ACCOUNT = "acct-operator"


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

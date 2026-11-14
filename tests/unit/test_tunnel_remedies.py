"""Inactive-tunnel reasons, remedy kinds and the post-sign-in no-spend run.

The field report: a teammate signed in with Radient, and the phone said his
tunnel "isn't active" while every surface told him to sign in again — a remedy
that can never clear billing or a console state. These tests pin the reason the
connector names for each cause, the action (`remedy.kind`) each surface offers,
and the guards on the unattended run a verified sign-in starts.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import AsyncMock

import pytest

from local_operator.tunnels import config, gateway, install, report, state
from local_operator.tunnels.errors import RequestFailed, TunnelInactive
from local_operator.tunnels.service import (
    authorization_failure_reason,
    classify_failure,
    refine_inactive,
)

VALUE = {"tunnel_id": "synthetic", "credential_id": 17, "gateway_port": 4100}


def _api(quote, record=None):
    """A fake owner-pinned client answering `/billing` and the tunnel record."""
    api = AsyncMock()

    async def request(method, path=""):
        if path == "/billing":
            if isinstance(quote, BaseException):
                raise quote
            return quote
        if isinstance(record, BaseException):
            raise record
        return record if record is not None else {"enabled": True, "status": "suspended"}

    api.request.side_effect = request
    return api


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quote,record,expected",
    [
        ({"active": False, "eligible": False}, None, gateway.BILLING_INACTIVE),
        ({"eligible": False}, None, gateway.BILLING_INACTIVE),
        ({"active": True, "eligible": False}, None, gateway.BILLING_LAPSED),
        ({"active": True, "eligible": True}, {"enabled": False}, gateway.TUNNEL_DISABLED),
        ({"active": True, "eligible": True}, {"enabled": True}, gateway.TUNNEL_INACTIVE),
        # Billing unreadable: the bare status's reason stands, never "unknown".
        (ValueError("no billing"), {"enabled": True}, gateway.TUNNEL_INACTIVE),
    ],
)
async def test_refine_names_the_actual_blocker(quote, record, expected):
    assert await refine_inactive(_api(quote, record), VALUE, default=gateway.TUNNEL_INACTIVE) == (
        expected
    )


@pytest.mark.asyncio
async def test_refine_uses_the_connect_record_without_a_second_read():
    api = _api({"active": True, "eligible": True}, RuntimeError("must not be read"))
    reason = await refine_inactive(
        api, VALUE, default=gateway.TUNNEL_INACTIVE, record={"enabled": False}
    )
    assert reason == gateway.TUNNEL_DISABLED
    # Read-only: only GETs, and never the activation endpoint.
    assert all(call.args[0] == "GET" for call in api.request.await_args_list)


@pytest.mark.parametrize(
    "reason",
    [
        gateway.TUNNEL_INACTIVE,
        gateway.TUNNEL_DISABLED,
        gateway.BILLING_INACTIVE,
        gateway.BILLING_LAPSED,
    ],
)
def test_a_refined_refusal_is_transient_and_never_a_sign_in(reason):
    failure = TunnelInactive(reason)
    failure.__cause__ = RequestFailed("Radient tunnel request failed (HTTP 403).", status=403)
    verdict = classify_failure(failure)
    # Transient: each clears from outside this process (a top-up, a console
    # toggle), and the connector must pick that up at its 10 s floor.
    assert verdict.kind == "transient"
    assert verdict.reason == reason
    assert authorization_failure_reason(failure) == reason
    # Every surface's copy is keyed, names no command and no sign-in.
    for text in (verdict.detail, gateway.relay_detail(reason), gateway.reason_label(reason)):
        assert text and "lop " not in text and "sign in" not in text.lower()
    assert gateway.TERMINAL_REMEDY[reason] != gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED]


@pytest.mark.parametrize(
    "reason,kind",
    [
        (gateway.TUNNEL_INACTIVE, "run"),
        (gateway.TUNNEL_DISABLED, "open_console"),
        (gateway.BILLING_INACTIVE, "accept_quote"),
        (gateway.BILLING_LAPSED, "add_credit"),
        (gateway.UNREACHABLE, "wait_network"),
        (gateway.CONTROL_PLANE_TIMEOUT, "wait_network"),
        (gateway.SERVER_UNAVAILABLE, "wait_radient"),
        (gateway.LOGIN_REQUIRED, "sign_in"),
    ],
)
def test_the_phone_503_carries_the_remedy_kind(reason, kind):
    gate = gateway.Gateway.__new__(gateway.Gateway)
    gate.revoked = False
    gate.authorization_failure = reason
    body = gate.unavailable_body()
    assert body["reason"] == reason
    assert body["kind"] == kind
    assert body["detail"]


def test_every_reason_code_has_a_remedy_kind():
    codes = {
        value
        for name, value in vars(gateway).items()
        if name.isupper() and isinstance(value, str) and name in gateway.REASON_LABEL.keys()
    }
    assert set(gateway.REASON_LABEL) <= set(gateway.REMEDY_KIND) | {gateway.LOGIN_SUPERSEDED}
    del codes


@pytest.mark.parametrize(
    "reason,kind,command",
    [
        (gateway.UNREACHABLE, "wait_network", None),
        (gateway.SERVER_UNAVAILABLE, "wait_radient", None),
        (gateway.TUNNEL_INACTIVE, "run", "lop tunnel setup"),
        (gateway.BILLING_LAPSED, "add_credit", "lop tunnel setup"),
        (gateway.TUNNEL_DISABLED, "open_console", "lop tunnel setup"),
    ],
)
def test_a_retrying_connector_gets_a_remedy_kind(reason, kind, command):
    connector = {"state": "stopped", "reason": reason, "remedy": None}
    remedy = report.remedy({"stopped": False}, connector, {"state": "ok"})
    assert remedy is not None
    assert remedy["kind"] == kind
    assert remedy["command"] == command


def test_self_clearing_and_stopped_states_still_have_no_remedy():
    ok = {"state": "ok"}
    lease = {"state": "not serving", "reason": gateway.LEASE_PENDING, "remedy": None}
    assert report.remedy({"stopped": False}, lease, ok) is None
    inactive = {"state": "stopped", "reason": gateway.TUNNEL_INACTIVE, "remedy": None}
    assert report.remedy({"stopped": True}, inactive, ok) is None


# --- The post-sign-in run's guards (U9) ----------------------------------------


@pytest.fixture
def hook(tmp_path, monkeypatch):
    """A sandboxed store the guards treat as the real home, with Popen recorded."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_TUNNEL_NO_REARM", raising=False)
    monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda path: True)
    spawned: list[dict[str, object]] = []

    def popen(argv, **kwargs):
        spawned.append({"argv": argv, **kwargs})

    monkeypatch.setattr(install.subprocess, "Popen", popen)
    return tmp_path, spawned


def _configure(**extra):
    config.save({"credential_id": 17, "tunnel_id": "synthetic", "gateway_port": 4100, **extra})


def test_the_run_starts_detached_and_no_spend_when_phone_access_was_wanted(hook):
    _tmp, spawned = hook
    _configure()
    note = install.auto_setup_after_login(provider="radient", credential_id=17)
    assert note
    (call,) = spawned
    argv = call["argv"]
    assert argv[-7:] == [
        "tunnel",
        "setup",
        "apply",
        "--credential-id",
        "17",
        "--no-spend",
        "--json",
    ]
    assert "--no-spend" in argv and "--accept-monthly-price" not in argv
    assert call["start_new_session"] is True
    assert call["stdin"] is subprocess.DEVNULL


def test_an_interrupted_create_intent_also_counts_as_wanted(hook):
    tmp, spawned = hook
    (config.directory() / "create.json").parent.mkdir(parents=True, exist_ok=True)
    (config.directory() / "create.json").write_text("{}")
    assert install.auto_setup_after_login(provider="radient", credential_id=17)
    assert len(spawned) == 1
    del tmp


def test_a_deliberately_stopped_tunnel_stays_stopped(hook):
    """U9: `lop tunnel stop` is the operator's decision; a sign-in never undoes it."""
    _tmp, spawned = hook
    _configure(stopped=True)
    assert install.auto_setup_after_login(provider="radient", credential_id=17) == ""
    assert spawned == []


@pytest.mark.parametrize(
    "reason", [gateway.LOGIN_REQUIRED, gateway.LOCAL_PREREQUISITE, gateway.REENROLMENT_REQUIRED]
)
def test_a_parked_tunnel_is_left_to_its_person(hook, reason):
    """U9: a park waits on a named act (the re-arm owns the login park)."""
    _tmp, spawned = hook
    _configure()
    state.mark_parked(reason=reason, detail="synthetic", credential_id=17)
    assert install.auto_setup_after_login(provider="radient", credential_id=17) == ""
    assert spawned == []


@pytest.mark.parametrize(
    "case",
    ["no_phone_access", "other_provider", "switch_off", "opted_out", "sandbox_home"],
)
def test_the_run_declines_outside_its_guards(hook, monkeypatch, case):
    tmp, spawned = hook
    provider = "radient"
    if case != "no_phone_access":
        _configure()
    if case == "other_provider":
        provider = "openai"
    if case == "switch_off":
        (tmp / "config.yml").write_text("values:\n  tunnel.auto_setup_after_login: false\n")
        assert install.auto_setup_enabled() is False
    if case == "opted_out":
        monkeypatch.setenv("LOP_TUNNEL_NO_REARM", "1")
    if case == "sandbox_home":
        monkeypatch.setattr(install.supervisors, "config_lives_in_real_home", lambda path: False)
    assert install.auto_setup_after_login(provider=provider, credential_id=17) == ""
    assert spawned == []


def test_the_switch_defaults_on_and_reads_both_spellings(hook):
    tmp, _spawned = hook
    assert install.auto_setup_enabled() is True
    (tmp / "config.yml").write_text("values:\n  tunnel:\n    auto_setup_after_login: false\n")
    assert install.auto_setup_enabled() is False


@pytest.mark.parametrize(
    "verified,healed,note,expected",
    [
        (True, "", "", True),
        (False, "", "", False),  # a token refresh writes the row too
        (True, "healed", "", False),  # the heal already restarted the connector
        (True, "", "rearmed", False),  # so did the re-arm
    ],
)
def test_the_sign_in_hook_runs_after_heal_and_rearm_only(verified, healed, note, expected):
    from unittest.mock import Mock

    from local_operator.providers.auth_store import AuthStore

    fake = Mock()
    fake.heal_pin_after_login.return_value = healed
    fake.rearm_if_parked.return_value = note
    fake.auto_setup_after_login.return_value = ""
    AuthStore._rearm_off_thread(fake, "radient", 17, verified, None)
    assert fake.auto_setup_after_login.called is expected


# --- TUI (U12, first-sign-in offer) --------------------------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        ("lop login radient", "/login radient"),
        ("lop tunnel setup", "/mobile enable"),
        ("lop tunnel setup apply --credential-id 17", "/mobile enable"),
        ("lop tunnel setupx", "lop tunnel setupx"),
        ("lop tunnel status", "lop tunnel status"),
    ],
)
def test_the_tui_spells_setup_remedies_as_mobile_enable(command, expected):
    from local_operator.tui.app import tunnel_card_command

    assert tunnel_card_command(command) == expected


def test_the_park_card_never_prints_the_setup_shell_command():
    from local_operator.tui.app import tunnel_park_card

    card = tunnel_park_card(gateway.TUNNEL_INACTIVE, "lop tunnel setup")
    assert "/mobile enable" in card and "lop tunnel" not in card


def test_the_first_sign_in_offer_shows_once_per_credential(tmp_path, monkeypatch):
    from local_operator.tunnels import autosetup

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from types import SimpleNamespace

    rows = [SimpleNamespace(id=4, updated_at=10, credential_type="oauth")]

    class Store:
        def __init__(self, *a, **k):
            pass

        def list_credentials(self, provider):
            return rows

        def close(self):
            pass

    monkeypatch.setattr("local_operator.providers.auth_store.AuthStore", Store)
    assert autosetup.login_offer_due() == 4
    autosetup.record_login_offer(4)
    # Shown == dismissed for this credential: it never returns.
    assert autosetup.login_offer_due() is None
    # A NEW sign-in row may see it once.
    rows.append(SimpleNamespace(id=9, updated_at=20, credential_type="oauth"))
    assert autosetup.login_offer_due() == 9
    # A machine that already has phone access is reconnected, never offered.
    _configure()
    assert autosetup.login_offer_due() is None


def test_the_offer_copy_names_the_verb_and_the_price_promise():
    from local_operator.i18n.keys import mobile_setup
    from local_operator.i18n.text import tr

    text = tr("mobile.setup")(mobile_setup.login_offer())
    assert "/mobile enable" in text
    assert "price" in text and "lop " not in text


# --- GET /v1/desktop/tunnel (U10) ---------------------------------------------


@pytest.mark.asyncio
async def test_the_health_payload_says_reconnecting_while_the_run_holds_the_lock(
    tmp_path, monkeypatch
):
    from local_operator.config_writer import nonblocking_file_claim
    from local_operator.tunnels import autosetup

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _configure()
    monkeypatch.setattr(
        report,
        "connector_state",
        AsyncMock(
            return_value={"state": "stopped", "reason": gateway.TUNNEL_INACTIVE, "remedy": None}
        ),
    )
    monkeypatch.setattr(report, "login_verdict", AsyncMock(return_value={"state": "ok"}))
    before = await report.local_payload()
    assert before["reconnecting"] is False
    assert before["remedy"]["kind"] == "run"
    autosetup.write_journal({"state": "running"})
    with nonblocking_file_claim(config.directory() / "setup.lock"):
        during = await report.local_payload()
    assert during["reconnecting"] is True
    assert during["remedy"] == {"kind": "wait_reconnect", "command": None, "url": None}
    assert json.dumps(during)

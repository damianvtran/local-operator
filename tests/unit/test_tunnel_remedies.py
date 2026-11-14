"""Inactive-tunnel reasons, remedy kinds and the post-sign-in no-spend run.

The field report: a teammate signed in with Radient, and the phone said his
tunnel "isn't active" while every surface told him to sign in again — a remedy
that can never clear billing or a console state. These tests pin the reason the
connector names for each cause, the action (`remedy.kind`) each surface offers,
and the guards on the unattended run a verified sign-in starts.
"""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace
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
    # A new reason cannot ship without the action a graphical surface offers.
    assert set(gateway.REASON_LABEL) <= set(gateway.REMEDY_KIND)


def test_the_capability_flag_is_advertised():
    # The desktop gates every new kind on this key (absent => old handling).
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["tunnel_remedy_kinds"] == 1


@pytest.mark.parametrize(
    "reason,kind,command",
    [
        (gateway.UNREACHABLE, "wait_network", None),
        (gateway.SERVER_UNAVAILABLE, "wait_radient", None),
        (gateway.TUNNEL_INACTIVE, "run", "lop tunnel setup"),
        # Console kinds are a link: no setup command (#3945 round 1, Q8).
        (gateway.BILLING_LAPSED, "add_credit", None),
        (gateway.TUNNEL_DISABLED, "open_console", None),
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
        # A child handle the reaper thread can wait on (N2).
        return SimpleNamespace(wait=lambda: 0)

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
    autosetup.write_journal({"state": "running", "pid": os.getpid()})
    with nonblocking_file_claim(config.directory() / "setup.lock"):
        during = await report.local_payload()
    assert during["reconnecting"] is True
    assert during["remedy"] == {"kind": "wait_reconnect", "command": None, "url": None}
    assert json.dumps(during)


# --- #3945 round-1 remediation ------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quote,expected",
    [
        # R4: active + ineligible WITH credit is a re-acceptance (quote), not lapsed.
        ({"active": True, "eligible": False, "balance_usd": "4.00"}, gateway.BILLING_INACTIVE),
        ({"active": True, "eligible": False, "balance_usd": 0}, gateway.BILLING_LAPSED),
        ({"active": True, "eligible": False, "balance_usd": "-1"}, gateway.BILLING_LAPSED),
    ],
)
async def test_lapsed_needs_a_balance_that_is_not_positive(quote, expected):
    reason = await refine_inactive(_api(quote), VALUE, default=gateway.TUNNEL_INACTIVE)
    assert reason == expected


@pytest.mark.asyncio
async def test_a_console_switch_off_wins_over_lapsed_billing():
    """Q3: one precedence with `setup.observe` — disabled wins (open_console)."""
    lapsed = {"active": True, "eligible": False, "balance_usd": 0}
    for record in ({"enabled": False}, None):
        api = _api(lapsed, {"enabled": False})
        reason = await refine_inactive(api, VALUE, default=gateway.TUNNEL_INACTIVE, record=record)
        assert reason == gateway.TUNNEL_DISABLED


@pytest.mark.parametrize("status", [402, 403])
def test_a_bare_403_outside_connect_stays_an_authorization_refusal(status):
    """R2: only the `/connect` raise site (a `TunnelInactive`) is an inactive cause."""
    failure = RequestFailed(f"Radient tunnel request failed (HTTP {status}).", status=status)
    assert authorization_failure_reason(failure) == gateway.REFUSED
    assert classify_failure(failure).reason not in gateway.INACTIVE_REASONS


@pytest.mark.asyncio
async def test_the_refinement_is_cached_per_tunnel_and_trigger(tmp_path, monkeypatch):
    """R3: a retrying suspended connector adds no Radient reads within the TTL."""
    from local_operator.tunnels import service

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    api = _api({"active": True, "eligible": False, "balance_usd": 0}, {"enabled": True})
    for _ in range(3):
        reason = await service.refine_inactive_cached(api, VALUE, default=gateway.TUNNEL_INACTIVE)
        assert reason == gateway.BILLING_LAPSED
    assert api.request.await_count == 2  # one record GET + one /billing, once
    # A different trigger (402) re-refines at once.
    await service.refine_inactive_cached(api, VALUE, default=gateway.BILLING_INACTIVE)
    assert api.request.await_count == 4
    # Invalidated (successful connect / setup run): the next retry reads again.
    service.invalidate_refinement()
    await service.refine_inactive_cached(api, VALUE, default=gateway.BILLING_INACTIVE)
    assert api.request.await_count == 6
    # Expired past the TTL: read again.
    cache = config.directory() / service.REFINE_CACHE
    record = json.loads(cache.read_text())
    record["at"] -= service.REFINE_TTL_S
    cache.write_text(json.dumps(record))
    await service.refine_inactive_cached(api, VALUE, default=gateway.BILLING_INACTIVE)
    assert api.request.await_count == 8


@pytest.mark.asyncio
async def test_an_unreadable_refinement_is_not_cached(tmp_path, monkeypatch):
    from local_operator.tunnels import service

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    api = _api(ValueError("no billing"), {"enabled": True})
    assert (
        await service.refine_inactive_cached(api, VALUE, default=gateway.TUNNEL_INACTIVE)
        == gateway.TUNNEL_INACTIVE
    )
    assert not (config.directory() / service.REFINE_CACHE).exists()


def test_the_inactive_label_no_longer_says_not_active():
    label = gateway.reason_label(gateway.TUNNEL_INACTIVE)
    assert label == "paused by Radient — reconnecting"
    assert gateway.REASON_LABEL[gateway.TUNNEL_INACTIVE] == label


# --- The TUI card, keyed by remedy kind (Q2/D4) --------------------------------


@pytest.mark.parametrize(
    "reason,expected",
    [
        (gateway.TUNNEL_INACTIVE, "! /mobile enable — paused by Radient"),
        (gateway.BILLING_INACTIVE, "! /mobile enable — billing not activated"),
        (gateway.BILLING_LAPSED, "! Add Radient credit to resume phone access"),
        (gateway.TUNNEL_DISABLED, "! Turn phone access on in the Radient console"),
    ],
)
def test_the_card_is_keyed_by_kind_and_fits_50_cells(reason, expected):
    from rich.cells import cell_len

    from local_operator.tui.app import ICON_APPROVALS, tunnel_park_card

    card = tunnel_park_card(reason, gateway.TERMINAL_REMEDY[reason])
    assert card == expected.replace("!", ICON_APPROVALS, 1)
    assert cell_len(card) <= 50
    assert "lop " not in card


def test_a_waiting_kind_card_names_no_command():
    from local_operator.tui.app import tunnel_park_card

    card = tunnel_park_card(gateway.UNREACHABLE, "lop tunnel status")
    assert "lop " not in card and "/mobile" not in card


def _service_log(tmp_path, reason, *, at=None):
    import datetime

    stamp = datetime.datetime.fromtimestamp(at if at is not None else __import__("time").time())
    config.directory().mkdir(parents=True, exist_ok=True)
    (config.directory() / "service.log").write_text(
        f"{stamp.isoformat()} local_operator.tunnels.service: "
        f"connector stopped reason={reason}: Radient is not serving this tunnel.\n"
    )


@pytest.mark.parametrize("reason", list(gateway.INACTIVE_REASONS))
def test_a_current_inactive_exit_raises_the_card_without_a_park(tmp_path, monkeypatch, reason):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _configure()
    _service_log(tmp_path, reason)
    notice = report.retrying_notice()
    assert notice == {"reason": reason, "remedy": None}
    # The card renders it through the same kind-keyed path as a park would.
    from local_operator.tui.app import tunnel_park_card

    card = tunnel_park_card(reason, "")
    assert "lop " not in card


def test_state_mirrors_the_gateway_inactive_reasons():
    assert state.INACTIVE_REASONS == gateway.INACTIVE_REASONS


@pytest.mark.parametrize("case", ["wait_reason", "old_line", "stopped", "no_tunnel"])
def test_the_retrying_card_stays_quiet_outside_its_gates(tmp_path, monkeypatch, case):
    import time

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    if case != "no_tunnel":
        _configure(stopped=case == "stopped")
    reason = gateway.UNREACHABLE if case == "wait_reason" else gateway.TUNNEL_INACTIVE
    _service_log(tmp_path, reason, at=time.time() - 3600 if case == "old_line" else None)
    assert report.retrying_notice() is None


# --- auto-setup.log bound and child reaping (N2) -------------------------------


def test_the_auto_setup_log_is_rotated_at_the_request_log_cap(tmp_path):
    from local_operator.tunnels.request_log import FILE_MAX_BYTES

    log = tmp_path / "auto-setup.log"
    log.write_bytes(b"x" * FILE_MAX_BYTES)
    install._bound_auto_setup_log(log)
    assert not log.exists()
    assert (tmp_path / "auto-setup.log.1").stat().st_size == FILE_MAX_BYTES
    small = tmp_path / "small.log"
    small.write_bytes(b"y")
    install._bound_auto_setup_log(small)
    assert small.read_bytes() == b"y"


def test_the_spawned_run_is_reaped(hook, monkeypatch):
    import threading

    _configure()
    waited = threading.Event()

    class Child:
        def wait(self):
            waited.set()

    monkeypatch.setattr(install.subprocess, "Popen", lambda argv, **kw: Child())
    assert install.auto_setup_after_login(provider="radient", credential_id=17)
    assert waited.wait(5), "the child handle must be waited on (no zombie)"


def test_the_offer_accents_the_slash_command(monkeypatch):
    """D6: `/mobile enable` is drawn in the inline-code ink, the rest stays muted."""
    from rich.style import Style
    from rich.text import Text

    from local_operator.tui import theme as theme_mod
    from local_operator.tui.widgets.transcript import NoticeBlock

    text = "Reach this computer from the phone app: run /mobile enable. Price first."
    block = NoticeBlock(text, "note", fold_width=200, accent="/mobile enable")
    built = block._build(200)
    assert isinstance(built, Text)
    signal = theme_mod.semantic_color("signal").lower()
    accented = []
    for span in built.spans:
        color = span.style.color if isinstance(span.style, Style) else None
        if color is not None and color.triplet is not None and color.triplet.hex == signal:
            accented.append(built.plain[span.start : span.end])
    assert accented == ["/mobile enable"]
    plain = NoticeBlock(text, "note", fold_width=200)._build(200)
    assert isinstance(plain, Text) and plain.plain == built.plain

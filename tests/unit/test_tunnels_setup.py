"""Setup controls use synthetic owners/routes, never the user's live services."""

import argparse
import json
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.tunnels import api, config, setup
from local_operator.tunnels.arguments import add_parser


@pytest.fixture(scope="module")
def gateway_server():
    # Exercise the real report producer; only the HTTP health boundary is fake.
    payload = {"ok": True, "connected": True}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def rig(tmp_path, monkeypatch, gateway_server):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOP_TUNNEL_NO_REARM", "1")
    monkeypatch.setattr(setup, "credential_id", lambda selected=None: selected or 17)
    monkeypatch.setattr(
        setup, "usable_credential_id", lambda selected, owner_account=None: selected
    )
    monkeypatch.setattr(setup.supervisors, "supervisor", lambda: "launchctl")
    monkeypatch.setattr(setup, "_cloudflared_preflight", lambda path=None: "/synthetic/cloudflared")
    monkeypatch.setattr(setup.mobile_install, "health", lambda port: {"ok": True})
    monkeypatch.setattr(setup.mobile_install, "gate_closed", lambda port: True)
    monkeypatch.setattr(setup.report, "login_verdict", AsyncMock(return_value={"state": "ok"}))
    gateway_server[1].update(ok=True, connected=True)
    quote = {"eligible": True, "balance_usd": 1, "monthly_price_usd": "0.05"}
    record = {
        "id": "synthetic",
        "gateway_port": gateway_server[0].server_port,
        "status": "active",
        "harnesses": [
            {
                "id": "local-operator",
                "enabled": True,
                "port": 4098,
                "hostname": "exact.example.test",
            }
        ],
    }
    calls = []

    async def request(self, method, path="", **kwargs):
        calls.append((method, path))
        return quote if path == "/billing" else record

    monkeypatch.setattr(setup.RadientTunnels, "request", request)
    monkeypatch.setattr(
        setup.RadientTunnels, "account_id", AsyncMock(return_value="synthetic-owner")
    )
    return quote, record, calls


@pytest.mark.parametrize(
    "version,ok", [("2025.4.0", True), ("2026.1.0", True), ("2025.3.9", False), ("unknown", False)]
)
def test_cloudflared_minimum_version_even_with_token_file(monkeypatch, version, ok):
    from types import SimpleNamespace

    monkeypatch.setattr(setup, "cloudflared_binary", lambda configured: "/synthetic/cloudflared")
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=f"cloudflared version {version}"),
    )
    if ok:
        assert setup._cloudflared_preflight() == "/synthetic/cloudflared"
    else:
        with pytest.raises(ValueError):
            setup._cloudflared_preflight()


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("", ("status", None, None)),
        ("enable --credential-id 17 0.05", ("enable", "0.05", 17)),
        ("enable", ("enable", None, None)),
        ("status --credential-id 23", ("status", None, 23)),
    ],
)
def test_mobile_argument_grammar(arg, expected):
    assert setup.parse_mobile(arg) == expected


@pytest.mark.parametrize(
    "arg",
    [
        "enable NaN",
        "status 0.05",
        "enable --credential-id",
        "enable --credential-id 0",
        "enable --yes",
        "start --credential-id 17",
    ],
)
def test_mobile_grammar_rejects_invalid_or_implicit_mutations(arg):
    with pytest.raises(ValueError):
        setup.parse_mobile(arg)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("0.050", "0.05", True),
        ("0.04", "0.05", False),
        ("NaN", "0.05", False),
        (None, "0.05", False),
    ],
)
def test_price_comparison_uses_amount_not_string_spelling(left, right, expected):
    assert setup._same_price(left, right) is expected


@pytest.mark.asyncio
async def test_consent_shows_existing_masked_identity_account_and_price(rig, monkeypatch):
    from types import SimpleNamespace

    class Store:
        def close(self):
            pass

        def get_credential(self, selected):
            return SimpleNamespace(
                data={"email": "operator@example.test", "access": "PRIVATE_SENTINEL"}
            )

    monkeypatch.setattr(setup, "AuthStore", Store)
    rig[0]["eligible"] = False
    plan = await setup.observe(17)
    text = setup.text(plan, surface="tui")
    assert "o***@example.test" in text and "credential 17" in text
    assert "USD 0.05/month" in text and "does not purchase credit" in text
    assert "/mobile enable 0.05 --credential-id 17" in text
    assert "PRIVATE_SENTINEL" not in json.dumps(plan) + text
    assert "operator@" not in text


def test_failure_summary_is_stage_specific_without_duplicate_streamed_receipts():
    plan = {
        "step": "failed",
        "verified": False,
        "credential_id": 17,
        "identity_label": "f***@example.test",
        "receipts": ["Cloud tunnel reservation saved on this device."],
        "remedy": "developer fallback",
        "failure_stage": "installation",
        "billing_outcome": "activated",
        "reservation_outcome": "saved",
    }
    text = setup.text(plan, surface="tui", include_receipts=False)
    assert (
        "background connector installation" in text
        and "Billing: subscription activation completed" in text
    )
    assert "Cloud tunnel reservation saved on this device." not in text
    assert "retry with /mobile enable --credential-id 17" in text and "retry apply" not in text
    assert "developer fallback" not in text


def save(record, **extra):
    config.save(
        {
            "credential_id": 17,
            "tunnel_id": "synthetic",
            "record": record,
            "gateway_port": record["gateway_port"],
            **extra,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prerequisite", ["cloudflared", "supervisor"])
async def test_preflight_precedes_every_mutation(rig, monkeypatch, prerequisite):
    from local_operator.tunnels import cli

    if prerequisite == "cloudflared":

        def missing(path=None):
            raise ValueError("Install cloudflared 2025.4.0 or newer")

        monkeypatch.setattr(setup, "_cloudflared_preflight", missing)
    else:
        monkeypatch.setattr(setup.supervisors, "supervisor", lambda: None)
    mutation = AsyncMock()
    monkeypatch.setattr(cli, "dispatch", mutation)
    result = await setup.apply(accepted="0.05")
    assert result["step"] == prerequisite
    mutation.assert_not_awaited()
    assert all(method == "GET" for method, _ in rig[2])
    assert not (config.directory() / "create.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("sources", [False, True])
async def test_mobile_bundle_preflight_uses_real_nonbuilding_guard(
    rig, monkeypatch, tmp_path, sources
):
    from unittest.mock import Mock

    from local_operator.tunnels import cli

    web = tmp_path / "synthetic-web"
    web.mkdir()
    if sources:
        (web / "package.json").write_text("{}")
    monkeypatch.setattr(setup.mobile_install, "_WEB_DIR", web)
    monkeypatch.setattr(setup.mobile_install, "health", lambda port: None)
    monkeypatch.setattr(setup.mobile_install, "gate_closed", lambda port: False)
    build = Mock(side_effect=AssertionError("No implicit build/install in observation"))
    monkeypatch.setattr(setup.mobile_install, "_build_bundle", build)
    mutation = AsyncMock()
    monkeypatch.setattr(cli, "dispatch", mutation)
    rig[0]["eligible"] = False
    result = await setup.apply(accepted="0.05")
    assert result["step"] == "mobile_bundle" and not result["ready"]
    assert all(method == "GET" for method, _ in rig[2])
    assert not (config.directory() / "config.json").exists()
    mutation.assert_not_awaited()
    build.assert_not_called()


@pytest.mark.asyncio
async def test_healthy_origin_bypasses_mobile_bundle_preparation(rig, monkeypatch):
    from unittest.mock import Mock

    guard = Mock(side_effect=AssertionError("Healthy origin must not be installed"))
    monkeypatch.setattr(setup.mobile_install, "ensure_bundle", guard)
    assert (await setup.observe())["ready"]
    guard.assert_not_called()


@pytest.mark.asyncio
async def test_default_plan_never_mutates_even_with_consent_flags(rig, monkeypatch):
    from local_operator.tunnels import cli

    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    mutation = AsyncMock()
    monkeypatch.setattr(cli, "dispatch", mutation)
    args = parser.parse_args(["tunnel", "setup", "--accept-monthly-price", "0.05", "--json"])
    result = json.loads(await setup.command(args))
    assert result["step"] == "apply"
    mutation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [None, "0.04", "NaN", "Infinity", "bad"])
async def test_activation_requires_exact_current_quote(rig, monkeypatch, accepted):
    from local_operator.tunnels import cli

    rig[0]["eligible"] = False
    mutation = AsyncMock()
    monkeypatch.setattr(cli, "dispatch", mutation)
    result = await setup.apply(accepted=accepted)
    assert result["step"] == ("quote" if accepted is None else "quote_changed")
    assert result["activation_required"] is True
    mutation.assert_not_awaited()


@pytest.mark.asyncio
async def test_eligible_no_price_runs_then_verifies_exact_url(rig, monkeypatch):
    from local_operator.tunnels import cli

    async def dispatch(args):
        if args.tunnel_command == "create":
            assert args.accept_monthly_price is None
            save(rig[1])
        return "provider secret MUST NOT be included"

    runner = AsyncMock(side_effect=dispatch)
    monkeypatch.setattr(cli, "dispatch", runner)
    result = await setup.apply()
    assert result["verified"] is True
    assert result["url"] == "https://exact.example.test"
    assert [c.args[0].tunnel_command for c in runner.await_args_list] == ["create", "install"]
    assert "provider secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_partial_failure_preserves_receipts_and_redacts_error(rig, monkeypatch):
    from local_operator.tunnels import cli

    async def dispatch(args):
        if args.tunnel_command == "install":
            raise ValueError("provider bearer FAKE_SECRET")
        save(rig[1])
        return "FAKE_SECRET"

    monkeypatch.setattr(cli, "dispatch", dispatch)
    result = await setup.apply()
    assert result["step"] == "failed"
    assert result["receipts"] == ["Cloud tunnel reservation saved on this device."]
    assert "FAKE_SECRET" not in json.dumps(result)


@pytest.mark.asyncio
async def test_fresh_account_and_cross_account_route_fail_closed(rig, monkeypatch):
    save(rig[1])
    monkeypatch.setattr(setup.RadientTunnels, "account_id", AsyncMock(return_value=None))
    assert (await setup.apply())["step"] == "account"
    monkeypatch.setattr(
        setup.RadientTunnels, "account_id", AsyncMock(return_value="different-owner")
    )
    rig[1]["owner_account_id"] = "synthetic-owner"
    result = await setup.apply()
    assert result["step"] == "owner_choice" and result["url"] is None


@pytest.mark.asyncio
async def test_activation_receipt_survives_failed_reservation(rig, monkeypatch):
    from local_operator.tunnels import cli

    rig[0]["eligible"] = False
    calls = []

    async def dispatch(args):
        calls.append(args.tunnel_command)
        if args.tunnel_command == "activate":
            assert args.credential_id == 17 and args.accept_monthly_price == "0.05"
            return "provider body is not a receipt"
        raise ValueError("reservation failed FAKE_SECRET")

    monkeypatch.setattr(cli, "dispatch", dispatch)
    result = await setup.apply(accepted="0.05")
    assert calls == ["activate", "create"]
    assert result["receipts"] == ["Billing activation completed at the accepted monthly price."]
    assert result["activation_required"] is False
    assert "FAKE_SECRET" not in json.dumps(result)


@pytest.mark.asyncio
async def test_healthy_existing_is_noop(rig, monkeypatch):
    from local_operator.tunnels import cli

    save(rig[1])
    mutation = AsyncMock()
    monkeypatch.setattr(cli, "dispatch", mutation)
    before = (config.directory() / "config.json").read_bytes()
    result = await setup.apply()
    assert result["step"] == "complete" and result["verified"]
    assert (config.directory() / "config.json").read_bytes() == before
    mutation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["cloud", "connector", "mobile", "gateway", "url"])
async def test_final_verification_fails_closed(rig, monkeypatch, gate, gateway_server):
    if gate == "cloud":
        rig[1]["status"] = "paused"
    elif gate == "connector":
        gateway_server[1]["connected"] = False
    elif gate == "mobile":
        monkeypatch.setattr(setup.mobile_install, "gate_closed", lambda port: False)
    elif gate == "gateway":
        monkeypatch.setattr(setup.install, "gateway_answers", lambda: False)
    else:
        rig[1]["harnesses"][0]["hostname"] = "user:FAKE_SECRET@bad.test/path"
    save(rig[1])
    result = await setup.observe()
    assert not result["verified"] and result["url"] is None
    assert "FAKE_SECRET" not in json.dumps(result)


@pytest.mark.asyncio
async def test_stopped_and_different_owner_never_repoint(rig, monkeypatch):
    save(rig[1], stopped=True)
    assert (await setup.apply())["step"] == "stopped"
    assert (await setup.apply(99))["step"] == "owner_choice"
    monkeypatch.setattr(setup, "usable_credential_id", lambda selected, owner_account=None: None)
    assert (await setup.apply())["step"] == "owner_missing"
    assert rig[2] == []


@pytest.mark.asyncio
async def test_ambiguous_owner_requires_selection(rig, monkeypatch):
    def ambiguous(selected):
        if selected is None:
            raise ValueError("multiple owners")
        return selected

    monkeypatch.setattr(setup, "credential_id", ambiguous)
    assert (await setup.observe())["step"] == "login"
    assert (await setup.observe(23))["credential_id"] == 23


@pytest.mark.asyncio
async def test_zero_credit_refuses_new_setup_but_preserves_pending_resume(rig, monkeypatch):
    from local_operator.tunnels import cli

    rig[0]["balance_usd"] = 0
    assert (await setup.apply())["step"] == "balance"
    payload = {
        "name": "original-name",
        "device_id": "original-device",
        "gateway_port": 4999,
        "harnesses": [{"id": "local-operator", "enabled": True, "port": 4998}],
    }
    config.directory().mkdir(parents=True, exist_ok=True)
    (config.directory() / "create.json").write_text(
        json.dumps({"credential_id": 17, "key": "same-key", "payload": payload})
    )

    async def dispatch(args):
        if args.tunnel_command == "create":
            assert args.name == "original-name" and args.gateway_port == 4999
            assert args.mobile_port == 4998
            assert json.loads((config.directory() / "create.json").read_text())["key"] == "same-key"
            save(rig[1])
        else:
            raise ValueError("stop synthetic install")
        return "saved"

    monkeypatch.setattr(cli, "dispatch", dispatch)
    assert (await setup.apply())["receipts"] == ["Cloud tunnel reservation saved on this device."]


# --- issue #3520: a dead or gone pin with a live same-account successor --------
#
# The 2026-11-06 incident, seen from the setup plan. `tunnel/config.json` pins a
# credential row whose grant the IdP refused (and whose legacy identity key meant
# a later sign-in landed on a NEW row), and the plan told the operator to sign in
# again — which writes another row and changes nothing — with the only executable
# remedy being a terminal command a desktop user cannot run.

ACCOUNT = "synthetic-owner"
#: The REAL verdict, captured before `rig` stubs it: whether the plan still says
#: `login` is the whole regression, so the plan must be built from the real one.
REAL_LOGIN_VERDICT = setup.report.login_verdict
REPOINT_REMEDY = (
    "A newer sign-in to the same Radient account is already on this device. "
    "Reconnect phone access to that sign-in — same account, same phone address."
)


def _dead_legacy_pin(account: str = ACCOUNT) -> int:
    """The pinned row exactly as the incident found it (dead grant, legacy key)."""
    from tests.unit.test_tunnel_pin_heal import _seed_dead_legacy_pin

    return _seed_dead_legacy_pin(account=account)


def _live_row(account: str = ACCOUNT) -> int:
    """The row a VERIFIED sign-in to `account` writes: same hub, live grant."""
    with closing(AuthStore()) as store:
        return store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": account,
                "api_endpoint": api.TUNNEL_HUB,
                "access": "live-access-test-only",
                "refresh": "live-refresh-test-only",
                "expires": int(time.time() * 1000) + 3_600_000,
            },
        ).id


def _incident(rig, *, pin: int | None = None, account: str = ACCOUNT, **extra) -> tuple[int, int]:
    """A configured tunnel pinning a dead row, with a live same-account row beside it."""
    pin = _dead_legacy_pin(account) if pin is None else pin
    fresh = _live_row(account)
    config.save(
        {
            "tunnel_id": "synthetic",
            "credential_id": pin,
            "gateway_port": rig[1]["gateway_port"],
            "record": {**rig[1], "owner_account_id": account},
            # The two keys a bare `tunnel configure` drops (#3519): a re-point that
            # used it instead of `install._repoint_pin` would fail on them below.
            "cloudflared_path": "/synthetic/cloudflared",
            "mobile_password": "kept-test-only",
            **extra,
        }
    )
    return pin, fresh


def _serve(rig) -> None:
    """The cloud record names a routable harness hostname: the verified state."""
    rig[1]["harnesses"][0]["hostname"] = "exact.example.test"


def _unserve(rig) -> None:
    """Keep the record cloud-active while its route is not verifiable yet.

    Nothing about the pin: this is what makes the run reach the installation
    stage, which is the stage a `tunnel configure` would rewrite the config in.
    """
    rig[1]["harnesses"][0]["hostname"] = "not a host"


def _real_pin_proof(monkeypatch, seen: list[object] | None = None) -> None:
    """Opt out of the rig's synthetic pin predicates for the real local proof."""

    def usable(selected, *, owner_account=None):
        if seen is not None:
            seen.append(owner_account)
        return api.usable_credential_id(selected, owner_account=owner_account)

    monkeypatch.setattr(setup, "usable_credential_id", usable)
    monkeypatch.setattr(setup, "credential_id", api.credential_id)
    monkeypatch.setattr(setup.report, "login_verdict", REAL_LOGIN_VERDICT)


@pytest.mark.asyncio
async def test_dead_pin_with_same_account_successor_offers_the_repoint(rig, monkeypatch):
    """REGRESSION (#3520): the plan said `login`, whose remedy is a re-sign-in."""
    pin, fresh = _incident(rig)
    seen: list[object] = []
    _real_pin_proof(monkeypatch, seen)

    plan = await setup.observe()

    assert plan["step"] == "repoint", plan
    assert plan["ready"] is True and plan["verified"] is False
    assert plan["credential_id"] == fresh
    assert plan["previous_credential_id"] == pin
    assert plan["identity_label"]
    assert plan["remedy"] == REPOINT_REMEDY
    # The cached owner must reach the proof: it is the only same-account evidence
    # a GONE pin leaves, so forgetting it here would regress exactly that case.
    assert seen and seen[0] == ACCOUNT
    # Nothing is claimed that this plan did not read.
    assert "cloud_active" not in plan and "monthly_price_usd" not in plan


@pytest.mark.asyncio
async def test_desktop_poll_and_fresh_plan_agree_and_the_poll_writes_nothing(rig, monkeypatch):
    pin, fresh = _incident(rig)
    _real_pin_proof(monkeypatch)
    before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in config.directory().iterdir()
    }

    local = await setup.observe_local()
    fresh_plan = await setup.observe()

    # The poll is what the desktop renders while the fresh plan runs: a different
    # verdict here is a visible flicker between `owner_missing` and `repoint`.
    assert (local["step"], local["ready"], local["credential_id"]) == ("repoint", True, fresh)
    assert (fresh_plan["step"], fresh_plan["ready"], fresh_plan["credential_id"]) == (
        "repoint",
        True,
        fresh,
    )
    assert local["previous_credential_id"] == fresh_plan["previous_credential_id"] == pin
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in config.directory().iterdir()}
    assert after == before, "the poll is metadata-only: no auth DB, WAL, schema or config write"


@pytest.mark.asyncio
async def test_gone_pin_keeps_its_cached_owner_as_the_evidence(rig, monkeypatch):
    """The pin's row is deleted; only `record.owner_account_id` proves the account."""
    _pin, fresh = _incident(rig, pin=4242)
    _real_pin_proof(monkeypatch)

    assert (await setup.observe())["step"] == "repoint"
    assert (await setup.observe_local())["credential_id"] == fresh


@pytest.mark.asyncio
async def test_the_run_posting_the_successor_is_not_an_account_switch(rig, monkeypatch):
    """The desktop's Run posts the id the plan named — the successor, not the pin."""
    _pin, fresh = _incident(rig)
    _real_pin_proof(monkeypatch)
    with closing(AuthStore()) as store:
        unrelated = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": "someone-else",
                "api_endpoint": api.TUNNEL_HUB,
                "access": "a",
                "refresh": "r",
                "expires": int(time.time() * 1000) + 3_600_000,
            },
        ).id

    assert (await setup.observe(fresh))["step"] == "repoint"
    assert (await setup.observe_local(fresh))["step"] == "repoint"
    # An id that is neither the pin nor its successor is still a refusal.
    assert (await setup.observe(unrelated))["step"] == "owner_choice"


@pytest.mark.asyncio
async def test_a_stopped_tunnel_reports_stopped_not_a_repoint(rig, monkeypatch):
    """`_repoint_pin` declines a stopped tunnel: never offer what it would refuse."""
    _pin, fresh = _incident(rig, stopped=True)
    _real_pin_proof(monkeypatch)

    for plan in (await setup.observe(), await setup.observe_local()):
        assert plan["step"] == "stopped", plan
        assert plan["ready"] is False
        # The identity line still names the row a resumed connector will use.
        assert plan["credential_id"] == fresh


@pytest.mark.asyncio
async def test_a_pending_create_uses_the_successor_without_a_repoint(rig, monkeypatch):
    """No tunnel config means nothing to re-point, and observation never writes."""
    pin = _dead_legacy_pin()
    fresh = _live_row()
    config.directory().mkdir(parents=True, exist_ok=True)
    (config.directory() / "create.json").write_text(
        json.dumps(
            {
                "credential_id": pin,
                "key": "pending-key",
                "payload": {
                    "name": "pending-name",
                    "device_id": "pending-device",
                    "gateway_port": 4999,
                    "harnesses": [{"id": "local-operator", "enabled": True, "port": 4998}],
                },
            }
        )
    )
    _real_pin_proof(monkeypatch)

    plan = await setup.observe()
    poll = await setup.observe_local()

    assert plan["credential_id"] == fresh and plan["step"] == "apply"
    assert poll["credential_id"] == fresh and poll["step"] == "quote_unread"
    assert not (config.directory() / "config.json").exists()


@pytest.mark.asyncio
async def test_a_different_account_row_is_never_a_successor(rig, monkeypatch):
    _dead = _dead_legacy_pin()
    _live_row("someone-else")
    config.save(
        {
            "tunnel_id": "synthetic",
            "credential_id": _dead,
            "record": {**rig[1], "owner_account_id": ACCOUNT},
        }
    )
    _real_pin_proof(monkeypatch)

    plan = await setup.observe()

    assert plan["step"] == "login" and plan["ready"] is False
    assert plan["credential_id"] == _dead
    assert "previous_credential_id" not in plan


@pytest.mark.asyncio
async def test_a_live_pin_is_never_second_guessed(rig, monkeypatch):
    live = _live_row()
    config.save(
        {
            "tunnel_id": "synthetic",
            "credential_id": live,
            "gateway_port": rig[1]["gateway_port"],
            "record": {**rig[1], "owner_account_id": ACCOUNT},
        }
    )
    _real_pin_proof(monkeypatch)

    plan = await setup.observe()

    assert plan["step"] != "repoint" and plan["credential_id"] == live
    assert "previous_credential_id" not in plan


@pytest.mark.asyncio
async def test_apply_repoints_the_pin_before_any_stage_reads_it(rig, monkeypatch):
    """The re-point is a ready step: Run must move the pin and then complete."""
    from local_operator.tunnels import cli

    _pin, fresh = _incident(rig)
    _real_pin_proof(monkeypatch)
    # The tunnel is cloud-active but not serving yet, so the run has to reach the
    # installation stage — while `tunnel configure` must NOT run for it (that is
    # what rewrites the config and drops `cloudflared_path`/`mobile_password`).
    _unserve(rig)
    calls: list[str] = []

    async def dispatch(args):
        calls.append(args.tunnel_command)
        _serve(rig)
        return "no provider body"

    monkeypatch.setattr(cli, "dispatch", dispatch)

    result = await setup.apply()

    assert calls == ["install"], "a cloud-active tunnel must not be reconfigured"
    saved = config.load()
    assert saved["credential_id"] == fresh
    assert saved["cloudflared_path"] == "/synthetic/cloudflared"
    assert saved["mobile_password"] == "kept-test-only"
    assert result["step"] == "complete" and result["verified"] is True
    assert result["receipts"] == [
        f"Reconnected phone access to this device’s newer sign-in for the same "
        f"account (credential {fresh}).",
        "Local connector installation completed.",
    ]
    assert result["receipt_steps"] == ["configuration", "installation"]


@pytest.mark.asyncio
async def test_a_declined_repoint_is_reported_honestly_and_stops(rig, monkeypatch):
    """The proof can vanish between the plan and the run: never continue silently."""
    from local_operator.tunnels import cli

    pin, _fresh = _incident(rig)
    _real_pin_proof(monkeypatch)
    plan = await setup.observe()
    assert plan["step"] == "repoint"
    mutation = AsyncMock()
    monkeypatch.setattr(cli, "dispatch", mutation)
    monkeypatch.setattr(api, "same_account_successor", lambda *a, **k: None)

    result = await setup._apply(None, None, initial_plan=plan)

    assert result["ready"] is False and result["verified"] is False
    assert result["step"] not in {"complete", "repoint"}
    assert config.load()["credential_id"] == pin, "the pin never moves without proof"
    mutation.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_later_failure_keeps_the_repoint_receipt_and_the_moved_pin(rig, monkeypatch):
    from local_operator.tunnels import cli

    _pin, fresh = _incident(rig)
    _real_pin_proof(monkeypatch)
    _unserve(rig)

    async def dispatch(args):
        raise ValueError("synthetic connector failure")

    monkeypatch.setattr(cli, "dispatch", dispatch)

    result = await setup.apply()

    assert result["step"] == "failed" and result["failure_stage"] == "installation"
    assert result["receipt_steps"] == ["configuration"]
    assert result["receipts"] == [
        f"Reconnected phone access to this device’s newer sign-in for the same "
        f"account (credential {fresh})."
    ]
    assert config.load()["credential_id"] == fresh, "a completed re-point is not undone"


@pytest.mark.asyncio
async def test_repoint_text_names_the_state_and_its_command(rig, monkeypatch):
    _pin, fresh = _incident(rig)
    _real_pin_proof(monkeypatch)
    plan = await setup.observe()

    cli_text = setup.text(plan, include_receipts=False)
    assert "Phone access needs reconnecting to this device’s newer sign-in" in cli_text
    assert REPOINT_REMEDY in cli_text
    assert f"lop tunnel setup apply --credential-id {fresh}" in cli_text

    tui_text = setup.text(plan, surface="tui", include_receipts=False)
    assert REPOINT_REMEDY in tui_text, "the TUI token rewrites must not mangle this copy"
    assert f"/mobile enable --credential-id {fresh}" in tui_text

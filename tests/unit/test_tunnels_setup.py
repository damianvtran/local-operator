"""Setup controls use synthetic owners/routes, never the user's live services."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock

import pytest

from local_operator.tunnels import config, setup
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
    monkeypatch.setattr(setup, "usable_credential_id", lambda selected: selected)
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
    monkeypatch.setattr(setup, "usable_credential_id", lambda selected: None)
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

"""Derived mobile setup: existing records are the journal, never a second store.

Observation cannot re-arm services (unlike legacy ``status``). Only explicit
apply advances the existing idempotent CLI operations; local prerequisites must
hold before even billing activation, since a failed install cannot undo a cloud
reservation or the user's price acceptance. Authentication/terms and software
installation remain outside this runner's authority.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
from collections.abc import Callable
from contextlib import closing
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlsplit

import httpx

from local_operator import supervisors
from local_operator.i18n.keys import mobile_setup as ms
from local_operator.i18n.text import tr
from local_operator.mobile import install as mobile_install
from local_operator.providers.auth_store import AuthStore
from local_operator.tunnels import config, install, report
from local_operator.tunnels.api import (
    RadientTunnels,
    credential_id,
    usable_credential_id,
)
from local_operator.tunnels.service import cloudflared_binary, tunnel_path

_t = tr("mobile.setup")


def _plan(step: str, remedy: str, **fields: Any) -> dict[str, Any]:
    return {
        "step": step,
        "ready": False,
        "verified": False,
        "remedy": remedy,
        "receipts": [],
        "url": None,
        **fields,
    }


def _price(value: object) -> str | None:
    # Provider responses are not transcript-safe. Only finite nonnegative money
    # crosses the setup contract; arbitrary response fields and errors do not.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return str(amount) if amount.is_finite() and amount >= 0 else None


def _cloudflared_preflight(configured: str | None = None) -> str:
    # Existing connector capability probe protects --token-file. Setup also
    # binds the documented minimum version instead of trusting feature text
    # from an older/backported binary as permission to mutate cloud resources.
    try:
        binary = cloudflared_binary(configured)
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cloudflared prerequisite unavailable") from exc
    match = re.search(r"cloudflared version (\d+)\.(\d+)\.(\d+)", result.stdout)
    if result.returncode or not match or tuple(map(int, match.groups())) < (2025, 4, 0):
        raise ValueError("cloudflared minimum version is 2025.4.0")
    return binary


def _mobile_url(record: dict[str, Any]) -> str | None:
    for harness in record.get("harnesses", []):
        if harness.get("id") == "local-operator" and harness.get("enabled"):
            hostname = harness.get("hostname")
            if not isinstance(hostname, str) or not hostname:
                continue
            if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", hostname):
                continue
            parsed = urlsplit("https://" + hostname)
            if (
                parsed.hostname == hostname
                and parsed.username is None
                and not parsed.path
                and not parsed.query
                and not parsed.fragment
            ):
                return "https://" + hostname
    return None


async def observe(selected: int | None = None) -> dict[str, Any]:
    """Read fresh owner/billing/route and local health without service writes."""
    from local_operator.tunnels.cli import _positive_balance

    value: dict[str, Any] = {}
    if (config.directory() / "config.json").exists():
        value = config.load()
    pending_path = config.directory() / "create.json"
    pending = json.loads(pending_path.read_text()) if pending_path.exists() else {}
    pin = value.get("credential_id") if value else pending.get("credential_id")
    if pin is not None and usable_credential_id(pin) is None:
        return _plan("owner_missing", _t(ms.owner_missing()))
    if pin is not None and selected is not None and selected != pin:
        return _plan("owner_choice", _t(ms.owner_mismatch()))
    try:
        selected = credential_id(selected if selected is not None else pin)
    except ValueError:
        with closing(AuthStore()) as store:
            choices = [
                row.id
                for row in store.list_credentials("radient")
                if row.credential_type == "oauth"
            ]
        return _plan(
            "owner_choice" if len(choices) > 1 else "login",
            _t(ms.choose_login()),
            credential_choices=choices,
        )
    fields: dict[str, Any] = {"credential_id": selected, "configured": bool(value)}
    if value.get("stopped"):
        return _plan("stopped", _t(ms.stopped()), **fields)
    if value:
        login = await report.login_verdict(value)
        if login.get("state") != "ok":
            return _plan("login", _t(ms.login()), **fields)
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            api = RadientTunnels(selected, client)
            account = await api.account_id()
            quote = await api.request("GET", "/billing")
            record = await api.request("GET", tunnel_path(value)) if value else {}
    except (ValueError, OSError, httpx.HTTPError):
        return _plan("account", _t(ms.account_failed()), **fields)
    if not account or not isinstance(quote, dict) or not isinstance(record, dict):
        return _plan("account", _t(ms.invalid_response()), **fields)
    old_record = value.get("record")
    old_owner = old_record.get("owner_account_id") if isinstance(old_record, dict) else None
    owner = record.get("owner_account_id")
    if (old_owner is not None and old_owner != account) or (owner is not None and owner != account):
        return _plan("owner_choice", _t(ms.owner_mismatch()), **fields)
    if value and (
        not isinstance(record.get("harnesses"), list)
        or any(not isinstance(h, dict) for h in record["harnesses"])
    ):
        return _plan("account", _t(ms.invalid_response()), **fields)
    fields.update(
        account_valid=True,
        positive_balance=_positive_balance(quote),
        activation_required=quote.get("eligible") is not True,
        monthly_price_usd=_price(quote.get("monthly_price_usd")),
        cloud_active=record.get("status") == "active",
    )
    # A healthy configuration is observed, never configured again. Cloud active
    # alone is insufficient: the closed origin gate and actual connector matter.
    if value:
        connector = await report.connector_state(value)
        harness = next(
            (
                h
                for h in record.get("harnesses", [])
                if h.get("id") == "local-operator" and h.get("enabled")
            ),
            None,
        )
        port = harness.get("port") if harness else None
        if isinstance(port, int) and not isinstance(port, bool):
            healthy, closed = await asyncio.to_thread(
                lambda: (mobile_install.health(port), mobile_install.gate_closed(port))
            )
        else:
            healthy, closed = None, False
        url = _mobile_url(record)
        gateway_healthy = await asyncio.to_thread(install.gateway_answers)
        if (
            connector.get("state") == "running"
            and record.get("status") == "active"
            and gateway_healthy
            and healthy is not None
            and closed
            and url
        ):
            return _plan(
                "complete", _t(ms.complete()), ready=True, verified=True, url=url, **fields
            )
    # An interrupted create reuses its reservation rather than treating renewal
    # as a new enrollment. It still needs a fresh valid account and quote.
    if not fields["positive_balance"] and not pending:
        return _plan("balance", _t(ms.balance()), **fields)
    try:
        await asyncio.to_thread(_cloudflared_preflight, value.get("cloudflared_path"))
    except (ValueError, OSError):
        return _plan("cloudflared", _t(ms.cloudflared()), **fields)
    if supervisors.supervisor() is None:
        return _plan("supervisor", install.NO_SUPERVISOR_ERROR, **fields)
    if fields["activation_required"]:
        if fields["monthly_price_usd"] is None:
            return _plan("quote", _t(ms.price_unavailable()), **fields)
        command = (
            f"lop tunnel setup apply --credential-id {selected} "
            f"--accept-monthly-price {fields['monthly_price_usd']}"
        )
        return _plan("quote", _t(ms.quote(command=command)), ready=True, **fields)
    command = f"lop tunnel setup apply --credential-id {selected}"
    return _plan("apply", _t(ms.apply(command=command)), ready=True, **fields)


async def apply(
    selected: int | None = None,
    accepted: str | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Advance forward steps, retaining each completed receipt even on failure."""
    from local_operator.tunnels.arguments import add_parser
    from local_operator.tunnels.cli import dispatch

    plan = await observe(selected)
    if plan["verified"] or not plan["ready"]:
        return plan
    if plan.get("activation_required") and (
        _price(accepted) is None or Decimal(str(accepted)) != Decimal(plan["monthly_price_usd"])
    ):
        return {**plan, "ready": False}
    selected = plan["credential_id"]
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipts: list[str] = []

    def receipt(message: str) -> None:
        receipts.append(message)
        if progress:
            progress(message)

    try:
        if plan.get("activation_required"):
            # Keep activation as its own receipt: a failed reservation cannot
            # undo billing that the user already accepted and the server applied.
            args = [
                "tunnel",
                "activate",
                "--credential-id",
                str(selected),
                "--accept-monthly-price",
                str(accepted),
            ]
            await dispatch(parser.parse_args(args))
            receipt(_t(ms.billing_activated()))
            plan = {**plan, "activation_required": False}
        value = config.load() if plan["configured"] else {}
        if not value:
            # Reuse the ORIGINAL payload as well as the existing key: hostname
            # changes between attempts must not make a lost response unresumable.
            args = ["tunnel", "create", "--credential-id", str(selected)]
            pending = config.directory() / "create.json"
            if pending.exists():
                payload = json.loads(pending.read_text())["payload"]
                args += ["--name", payload["name"], "--gateway-port", str(payload["gateway_port"])]
                for name, flag in (("local-operator", "mobile"), ("opencode", "opencode")):
                    harness = next((h for h in payload["harnesses"] if h["id"] == name), None)
                    if harness:
                        args += [f"--{flag}-port", str(harness["port"])]
                        if not harness["enabled"]:
                            args += [f"--no-{flag}"]
            if accepted is not None:
                args += ["--accept-monthly-price", accepted]
            await dispatch(parser.parse_args(args))
            receipt(_t(ms.reserved()))
        elif plan.get("activation_required") or not plan.get("cloud_active"):
            args = ["tunnel", "configure", "--enable", "--credential-id", str(selected)]
            if accepted is not None:
                args += ["--accept-monthly-price", accepted]
            await dispatch(parser.parse_args(args))
            receipt(_t(ms.activated()))
        await dispatch(parser.parse_args(["tunnel", "install"]))
        receipt(_t(ms.installed()))
        # Startup is asynchronous. Bound observation without inventing a URL or
        # turning a starting connector into a success claim.
        for attempt in range(5):
            plan = await observe(selected)
            if plan["verified"]:
                break
            if attempt < 4:
                await asyncio.sleep(1)
        if not plan["verified"]:
            plan = {**plan, "step": "verification", "ready": False, "remedy": _t(ms.verification())}
    except (ValueError, OSError, httpx.HTTPError):
        # Error bodies, saved records and connector tokens never enter a receipt.
        # Earlier successes remain visible so retry is not mistaken for rollback.
        plan = {
            **plan,
            "step": "failed",
            "ready": False,
            "verified": False,
            "url": None,
            "remedy": _t(ms.failed()),
        }
    return {**plan, "receipts": receipts}


def text(plan: dict[str, Any], *, surface: str = "cli") -> str:
    lines = [_t(ms.title(step=str(plan["step"])))]
    if plan.get("monthly_price_usd") is not None and plan.get("activation_required"):
        lines.append(_t(ms.price(amount=str(plan["monthly_price_usd"]))))
    lines.extend(plan["receipts"])
    remedy = plan["remedy"]
    if surface == "tui":
        # Keep the shared state/consent binding; only command spelling changes.
        # A TUI user never needs to copy a generated acceptance into a terminal.
        remedy = remedy.replace("lop tunnel setup apply", "/mobile enable")
        remedy = remedy.replace("--accept-monthly-price ", "")
        remedy = remedy.replace("lop tunnel setup --json", "/mobile status")
        remedy = remedy.replace("lop tunnel setup", "/mobile status")
        remedy = remedy.replace("lop login-status", "/login status")
        remedy = remedy.replace("lop login radient", "/login radient")
        remedy = remedy.replace("lop tunnel start", "/mobile start")
    lines.append(remedy)
    if plan.get("credential_choices"):
        lines.append(_t(ms.credential_ids(ids=", ".join(map(str, plan["credential_choices"])))))
    if plan["verified"]:
        lines.append(_t(ms.verified_url(url=str(plan["url"]))))
    return "\n".join(lines)


def parse_mobile(arg: str) -> tuple[str, str | None, int | None]:
    """Small slash grammar shared by tests and the app, with no implicit apply."""
    pieces = arg.split()
    action = pieces.pop(0) if pieces else "status"
    if action not in {"status", "enable", "billing", "start", "stop"}:
        raise ValueError("Unknown mobile action")
    selected = None
    if "--credential-id" in pieces:
        index = pieces.index("--credential-id")
        if index + 1 >= len(pieces):
            raise ValueError("Missing credential id")
        selected = int(pieces[index + 1])
        if selected <= 0 or action in {"start", "stop"}:
            raise ValueError("Invalid credential selection")
        del pieces[index : index + 2]
    if len(pieces) > 1 or (pieces and action != "enable"):
        raise ValueError("Unexpected mobile arguments")
    accepted = pieces[0] if pieces else None
    if accepted is not None and _price(accepted) is None:
        raise ValueError("Invalid monthly price")
    return action, accepted, selected


async def command(args: argparse.Namespace) -> str:
    plan = await (
        apply(args.credential_id, args.accept_monthly_price)
        if args.setup_action == "apply"
        else observe(args.credential_id)
    )
    return json.dumps(plan, allow_nan=False) if args.json else text(plan)

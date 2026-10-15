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
import threading
from collections.abc import Callable
from contextlib import closing
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlsplit

import httpx

from local_operator import supervisors
from local_operator.config_writer import nonblocking_file_claim
from local_operator.i18n.keys import mobile_setup as ms
from local_operator.i18n.text import tr
from local_operator.mobile import install as mobile_install
from local_operator.network.credentials.offers import mask_label
from local_operator.owned_tasks import join_owned
from local_operator.providers.auth_store import AuthStore, credential_identity
from local_operator.tunnels import config, install, report
from local_operator.tunnels.api import (
    RadientTunnels,
    credential_id,
    usable_credential_id,
)
from local_operator.tunnels.service import cloudflared_binary, tunnel_path

_t = tr("mobile.setup")

# All in-process owners share this guard; the kernel claim below also excludes
# a separate CLI process against the same journal. Never keyed by credential:
# two accounts still contend for one machine's connector/configuration.
_APPLY_GUARD = threading.Lock()


class SetupInProgress(ValueError):
    """Another owner still holds this machine's mutation journal."""


async def apply(
    selected: int | None = None,
    accepted: str | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    plan = await observe(selected)
    if (
        plan["verified"]
        or not plan["ready"]
        or (
            plan.get("activation_required") and not _same_price(accepted, plan["monthly_price_usd"])
        )
    ):
        # Read-only/healthy cases must not create a lease file as a hidden write.
        return await _apply(selected, accepted, progress=progress, initial_plan=plan)
    if not _APPLY_GUARD.acquire(blocking=False):
        raise SetupInProgress(_t(ms.in_progress()))
    try:
        with nonblocking_file_claim(config.directory() / "setup.lock") as acquired:
            if not acquired:
                raise SetupInProgress(_t(ms.in_progress()))
            operation = asyncio.create_task(_apply(selected, accepted, progress=progress))
            # Reobserve under the claim, and drain even a repeatedly cancelled
            # response before its lease can be released around blocking install.
            return await join_owned(operation)
    finally:
        _APPLY_GUARD.release()


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


def _same_price(accepted: object, current: object) -> bool:
    left, right = _price(accepted), _price(current)
    return left is not None and right is not None and Decimal(left) == Decimal(right)


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


def _identity_label(selected: int) -> str:
    # Share credential disclosure's whitelist and masking; never serialize a
    # login payload or the fresh /me response into a setup receipt.
    with closing(AuthStore()) as store:
        row = store.get_credential(selected)
        identity = credential_identity(row) if row is not None else None
    return mask_label(identity) if identity else _t(ms.identity_unavailable())


def _mobile_preflight(value: dict[str, Any], pending: dict[str, Any]) -> bool:
    # Mirror _prepare_mobile's healthy-origin skip, including its auth gate.
    # Otherwise require the installer's guard without a build, password write
    # or service start: source installs prepare mobile with normal tool approval
    # BEFORE a billing/reservation POST. The existing verifier is bounded.
    record = value.get("record") or pending.get("payload") or {}
    mobile = next((h for h in record.get("harnesses", []) if h.get("id") == "local-operator"), None)
    if mobile and not mobile.get("enabled"):
        return False
    port = mobile.get("port") if mobile else mobile_install.DEFAULT_PORT
    if mobile_install.health(port) and mobile_install.gate_closed(port):
        return True
    ok, _private_detail = mobile_install.ensure_bundle(build=False)
    return ok


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


def local_credential_choices() -> list[dict[str, Any]]:
    """Read-only identity whitelist shared by the local poll and transport."""
    with closing(AuthStore(read_only=True)) as store:
        rows = [r for r in store.list_credentials("radient") if r.credential_type == "oauth"]
    return [
        {
            "id": row.id,
            "identity_label": mask_label(credential_identity(row) or "")
            or _t(ms.identity_unavailable()),
        }
        for row in rows
    ]


async def observe_local(selected: int | None = None) -> dict[str, Any]:
    """Poll-on-open facts without upstream reads, refreshes or store writes.

    A cached active record is not a verified route. Even a connected loopback
    probe cannot supply fresh billing/account/cloud authority to a desktop poll.
    Metadata-only AuthStore is intentional: its constructor cannot create an
    auth database or mutate WAL/schema while merely opening this affordance.
    """
    value = config.load() if (config.directory() / "config.json").exists() else {}
    pending_path = config.directory() / "create.json"
    pending = json.loads(pending_path.read_text()) if pending_path.exists() else {}
    pin = value.get("credential_id") if value else pending.get("credential_id")
    choices = local_credential_choices()
    fields: dict[str, Any] = {
        "configured": bool(value),
        "credential_choices": choices,
        "credential_id": selected if selected is not None else pin,
    }
    if pin is not None and selected is not None and selected != pin:
        return _plan("owner_choice", _t(ms.owner_mismatch()), **fields)
    requested = selected if selected is not None else pin
    matches = [r for r in choices if requested is None or r["id"] == requested]
    if len(matches) != 1:
        return _plan(
            (
                "owner_missing"
                if pin is not None and not matches
                else "owner_choice" if len(matches) > 1 else "login"
            ),
            _t(ms.owner_missing()) if pin is not None and not matches else _t(ms.choose_login()),
            **fields,
        )
    fields.update(credential_id=matches[0]["id"], identity_label=matches[0]["identity_label"])
    if value.get("stopped"):
        return _plan("stopped", _t(ms.stopped()), **fields)
    if value:
        # A poll must remain loopback-only even with malformed local config;
        # do not interpolate an unchecked authority/userinfo string into a URL.
        value = {**value, "gateway_port": config.port(value.get("gateway_port"))}
        fields["connector"] = await report.connector_state(value)
    return _plan("quote_unread", _t(ms.quote_unread()), **fields)


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
            _t(ms.choose_login()) if len(choices) > 1 else _t(ms.login()),
            credential_choices=choices,
        )
    fields: dict[str, Any] = {
        "credential_id": selected,
        "configured": bool(value),
        "identity_label": _identity_label(selected),
    }
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
            connector.get("state") == "connected"
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
    if not await asyncio.to_thread(_mobile_preflight, value, pending):
        return _plan("mobile_bundle", _t(ms.mobile_bundle()), **fields)
    if fields["activation_required"]:
        if fields["monthly_price_usd"] is None:
            return _plan("quote", _t(ms.price_unavailable()), **fields)
        command = (
            f"lop tunnel setup apply --credential-id {selected} "
            f"--accept-monthly-price {fields['monthly_price_usd']}"
        )
        return _plan("quote", _t(ms.quote()), ready=True, apply_command=command, **fields)
    command = f"lop tunnel setup apply --credential-id {selected}"
    return _plan("apply", _t(ms.apply(command=command)), ready=True, **fields)


async def _apply(
    selected: int | None = None,
    accepted: str | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    initial_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance under an owned claim; blocked observations never take a claim."""
    from local_operator.tunnels.arguments import add_parser
    from local_operator.tunnels.cli import dispatch

    plan = initial_plan if initial_plan is not None else await observe(selected)
    if plan["verified"] or not plan["ready"]:
        return plan
    if plan.get("activation_required") and not _same_price(accepted, plan["monthly_price_usd"]):
        if accepted is None:
            return {**plan, "ready": False}
        return {
            **plan,
            "step": "quote_changed",
            "ready": False,
            "accepted_price_usd": _price(accepted),
            "remedy": _t(
                ms.quote_changed(
                    accepted=str(_price(accepted)), current=str(plan["monthly_price_usd"])
                )
            ),
        }
    selected = plan["credential_id"]
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    receipts: list[str] = []
    receipt_steps: list[str] = []
    stage = "prerequisites"
    billing = "not_activated" if plan.get("activation_required") else "already_active"
    reservation = "existing" if plan["configured"] else "not_requested"

    def phase(message: str) -> None:
        if progress:
            progress(message)

    def receipt(message: str) -> None:
        receipts.append(message)
        receipt_steps.append(stage)
        if progress:
            progress(message)

    try:
        if plan.get("activation_required"):
            stage = "billing"
            billing = "unconfirmed"
            phase(_t(ms.activating()))
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
            billing = "activated"
            plan = {**plan, "activation_required": False}
        value = config.load() if plan["configured"] else {}
        if not value:
            stage = "reservation"
            reservation = "unconfirmed"
            phase(_t(ms.reserving()))
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
            reservation = "saved"
        elif plan.get("activation_required") or not plan.get("cloud_active"):
            stage = "configuration"
            phase(_t(ms.configuring()))
            args = ["tunnel", "configure", "--enable", "--credential-id", str(selected)]
            if accepted is not None:
                args += ["--accept-monthly-price", accepted]
            await dispatch(parser.parse_args(args))
            receipt(_t(ms.activated()))
        stage = "installation"
        phase(_t(ms.installing()))
        await dispatch(parser.parse_args(["tunnel", "install"]))
        receipt(_t(ms.installed()))
        # Startup is asynchronous. Bound observation without inventing a URL or
        # turning a starting connector into a success claim.
        stage = "verification"
        phase(_t(ms.verifying()))
        for attempt in range(5):
            plan = await observe(selected)
            if plan["verified"]:
                break
            if attempt < 4:
                await asyncio.sleep(1)
        if not plan["verified"]:
            plan = {**plan, "step": "verification", "ready": False, "remedy": _t(ms.verification())}
    except (ValueError, OSError, httpx.HTTPError, KeyError, TypeError):
        # Error bodies, saved records and connector tokens never enter a receipt.
        # Earlier successes remain visible so retry is not mistaken for rollback.
        # A lost activation response is NOT proof that billing rolled back.
        # One fresh observation may confirm eligibility or a moved quote, without
        # carrying an arbitrary exception/provider response into the transcript.
        moved_quote = False
        if stage == "billing":
            fresh = await observe(selected)
            if fresh.get("account_valid"):
                moved_quote = (
                    accepted is not None
                    and fresh.get("activation_required")
                    and not _same_price(accepted, fresh.get("monthly_price_usd"))
                )
                if fresh.get("activation_required") is False:
                    billing = "activated"
                elif moved_quote:
                    billing = "not_activated"
                plan = fresh
        if stage == "reservation":
            if (config.directory() / "config.json").exists():
                reservation = "saved"
            elif not (config.directory() / "create.json").exists():
                reservation = "not_requested"
        plan = {
            **plan,
            "step": "quote_changed" if moved_quote else "failed",
            "accepted_price_usd": _price(accepted),
            "failure_stage": stage,
            "ready": False,
            "verified": False,
            "url": None,
            "remedy": (
                _t(
                    ms.quote_changed(
                        accepted=str(_price(accepted)), current=str(plan.get("monthly_price_usd"))
                    )
                )
                if moved_quote
                else _t(ms.failed())
            ),
        }
    return {
        **plan,
        "receipts": receipts,
        "receipt_steps": receipt_steps,
        "billing_outcome": billing,
        "reservation_outcome": reservation,
    }


def _apply_command(plan: dict[str, Any], surface: str) -> str:
    command = "/mobile enable" if surface == "tui" else "lop tunnel setup apply"
    if plan.get("activation_required") and plan.get("monthly_price_usd") is not None:
        amount = str(plan["monthly_price_usd"])
        command += f" {amount}" if surface == "tui" else f" --accept-monthly-price {amount}"
    if plan.get("credential_id") is not None:
        command += f" --credential-id {plan['credential_id']}"
    return command


def text(plan: dict[str, Any], *, surface: str = "cli", include_receipts: bool = True) -> str:
    titles = {
        "login": ms.title_login,
        "owner_choice": ms.title_owner_choice,
        "owner_missing": ms.title_owner_missing,
        "quote": ms.title_quote,
        "quote_changed": ms.title_quote_changed,
        "cloudflared": ms.title_cloudflared,
        "mobile_bundle": ms.title_mobile_bundle,
        "supervisor": ms.title_supervisor,
        "balance": ms.title_balance,
        "account": ms.title_account,
        "apply": ms.title_apply,
        "complete": ms.title_complete,
        "verification": ms.title_verification,
        "failed": ms.title_failed,
        "stopped": ms.title_stopped,
    }
    lines = [_t(titles.get(plan["step"], ms.title_failed)())]
    if plan.get("credential_id") is not None:
        lines.append(
            _t(
                ms.account(
                    identity=str(plan.get("identity_label") or _t(ms.identity_unavailable())),
                    id=str(plan["credential_id"]),
                )
            )
        )
    if plan.get("activation_required") and plan.get("monthly_price_usd") is not None:
        lines.append(_t(ms.price(amount=str(plan["monthly_price_usd"]))))
        if plan["step"] in {"quote", "quote_changed"}:
            lines.append(_t(ms.consent(amount=str(plan["monthly_price_usd"]))))
    if plan["verified"]:
        lines.append(_t(ms.verified_url(url=str(plan["url"]))))
    if include_receipts:
        lines.extend(plan["receipts"])
    stages = {
        "billing": ms.stage_billing,
        "reservation": ms.stage_reservation,
        "configuration": ms.stage_configuration,
        "installation": ms.stage_installation,
        "verification": ms.stage_verification,
        "prerequisites": ms.stage_prerequisites,
    }
    if plan.get("failure_stage") in stages:
        lines.append(_t(ms.failure_stage(stage=_t(stages[plan["failure_stage"]]()))))
    billing = {
        "not_activated": ms.billing_not_activated,
        "already_active": ms.billing_already_active,
        "activated": ms.billing_activated_status,
        "unconfirmed": ms.billing_unconfirmed,
    }
    reservation = {
        "saved": ms.reservation_saved,
        "existing": ms.reservation_existing,
        "unconfirmed": ms.reservation_unconfirmed,
    }
    if plan.get("billing_outcome") in billing:
        lines.append(_t(billing[plan["billing_outcome"]]()))
    if plan.get("reservation_outcome") in reservation:
        lines.append(_t(reservation[plan["reservation_outcome"]]()))
    remedy = plan["remedy"]
    if plan["step"] == "quote_changed":
        remedy = _t(
            ms.quote_changed(
                accepted=str(plan.get("accepted_price_usd")),
                current=str(plan.get("monthly_price_usd")),
            )
        )
    if surface == "tui":
        specific = {
            "supervisor": ms.supervisor_tui,
            "cloudflared": ms.cloudflared_tui,
            "mobile_bundle": ms.mobile_bundle_tui,
            "verification": ms.verification_tui,
            "complete": ms.complete_tui,
        }
        if plan["step"] in specific:
            remedy = _t(specific[plan["step"]]())
        elif plan["step"] == "failed":
            remedy = _t(ms.failed_tui(command=_apply_command(plan, surface)))
        elif plan["step"] == "login" and len(plan.get("credential_choices", [])) < 2:
            remedy = _t(ms.login_tui())
        # Only established command tokens are rewritten; no invented slash
        # equivalent for foreground serve or installing a background supervisor.
        remedy = remedy.replace("lop tunnel setup apply", "/mobile enable")
        remedy = remedy.replace("--accept-monthly-price ", "")
        remedy = remedy.replace("lop tunnel setup --json", "/mobile status")
        remedy = remedy.replace("lop tunnel setup", "/mobile status")
        remedy = remedy.replace("lop login-status", "/login status")
        remedy = remedy.replace("lop login radient", "/login radient")
        remedy = remedy.replace("lop tunnel start", "/mobile start")
    lines.append(remedy)
    if plan.get("activation_required") and plan["step"] in {"quote", "quote_changed"}:
        lines.append(_apply_command(plan, surface))
    if plan.get("credential_choices"):
        lines.append(_t(ms.credential_ids(ids=", ".join(map(str, plan["credential_choices"])))))
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

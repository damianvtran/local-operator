"""Small CLI surface, also used by /mobile; no permanent agent-tool schema."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

from local_operator.tunnels import config
from local_operator.tunnels.api import RadientTunnels, credential_id
from local_operator.tunnels.service import cloudflared_binary, tunnel_path


def _read_origin_auth(path: Path) -> dict[str, str]:
    if path.stat().st_mode & 0o077:
        raise ValueError("The OpenCode auth file must be private (chmod 600).")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) != {"username", "password"}:
        raise ValueError("OpenCode auth file must contain username and password only.")
    if any(not isinstance(v, str) or not v or "\n" in v or "\r" in v for v in data.values()):
        raise ValueError("OpenCode credentials must be nonempty single-line strings.")
    if ":" in data["username"]:
        raise ValueError("OpenCode username cannot contain a colon.")
    return data


def _harnesses(
    args: argparse.Namespace, previous: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    rows = {h["id"]: {k: h[k] for k in ("id", "enabled", "port")} for h in previous or []}
    if not rows:
        rows["local-operator"] = {"id": "local-operator", "enabled": True, "port": 4098}
    for name, prefix in (("local-operator", "mobile"), ("opencode", "opencode")):
        selected_port = getattr(args, prefix + "_port", None)
        if selected_port is not None:
            rows[name] = {"id": name, "enabled": True, "port": config.port(selected_port)}
        if getattr(args, "no_" + prefix, False) and name in rows:
            rows[name]["enabled"] = False
    if previous is None and not any(h["enabled"] for h in rows.values()):
        raise ValueError("Enable at least one harness.")
    return list(rows.values())


def _summary(record: dict[str, Any]) -> str:
    lines = [
        f"Tunnel: {record.get('id', 'not created')}",
        f"Status: {record.get('status', 'configured')}",
    ]
    for harness in record.get("harnesses", []):
        if harness.get("enabled") and harness.get("hostname"):
            lines.append(f"{harness['id']}: https://{harness['hostname']}")
    billing = record.get("billing")
    if "suspend" in str(record.get("status", "")) or (
        isinstance(billing, dict) and not billing.get("eligible")
    ):
        lines.append("Tunnel billing is suspended.")
        lines.append(
            "Add Radient credit and reactivate from "
            "https://console.radienthq.com/dashboard/tunnels."
        )
    return "\n".join(lines)


def _billing_summary(quote: dict[str, Any]) -> str:
    return (
        f"Monthly underlying cost: USD {quote['monthly_cost_usd']}\n"
        f"Monthly tunnel price: USD {quote['monthly_price_usd']}\n"
        f"Radient balance: USD {quote['balance_usd']}\n"
        f"Amount due: USD {quote['amount_due_usd']}\n"
        "Billing and credit: https://console.radienthq.com/dashboard/tunnels"
    )


def _positive_balance(quote: dict[str, Any]) -> bool:
    """Enrollment requires real credit; renewal still uses the cloud's -$1 floor.

    Missing/boolean/nonfinite amounts are unavailable, never a free-account
    shortcut. This reads the fresh owner-pinned quote, not the usage cache.
    """
    value = quote.get("balance_usd")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    try:
        balance = Decimal(str(value))
    except InvalidOperation:
        return False
    return balance.is_finite() and balance > 0


async def _ensure_billing(
    api: RadientTunnels, accepted: str | None, *, setup: bool = False
) -> dict[str, Any]:
    quote = await api.request("GET", "/billing")
    if setup and (not isinstance(quote, dict) or not _positive_balance(quote)):
        raise ValueError(
            "Tunnel setup requires a verified Radient balance above USD 0. "
            "Add credit at https://console.radienthq.com/dashboard/tunnels, then rerun "
            "lop tunnel billing --json with the same --credential-id."
        )
    if quote.get("eligible"):
        return quote
    if accepted is None:
        raise ValueError(
            _billing_summary(quote) + "\nAccept this quote with --accept-monthly-price <amount>."
        )
    try:
        amount = Decimal(accepted)
        expected = Decimal(str(quote["monthly_price_usd"]))
    except InvalidOperation:
        raise ValueError("Accepted monthly price must be a valid USD amount.") from None
    if not amount.is_finite() or amount != expected:
        raise ValueError(
            _billing_summary(quote) + "\nThe accepted price differs from the current quote."
        )
    result = await api.request(
        "POST", "/billing/activate", body={"accepted_monthly_price_usd": float(amount)}
    )
    if not result.get("eligible") or (setup and not _positive_balance(result)):
        raise ValueError(_billing_summary(result) + "\nAdd credit before activating the tunnel.")
    return result


def _prepare_mobile(value: dict[str, Any]) -> None:
    from local_operator.mobile import install as mobile_install

    mobile = next(
        (h for h in value["record"]["harnesses"] if h["id"] == "local-operator" and h["enabled"]),
        None,
    )
    if mobile is None:
        return
    if mobile_install.health(mobile["port"]) and mobile_install.gate_closed(mobile["port"]):
        return
    result = mobile_install.install(port=mobile["port"])
    if not result.get("ok"):
        raise ValueError(
            "Mobile relay installation failed. Run lop mobile install; "
            "on Linux run lop mobile serve first."
        )


async def dispatch(args: argparse.Namespace) -> str:
    from local_operator.tunnels import install

    action = args.tunnel_command
    if action in {"billing", "activate"}:
        async with httpx.AsyncClient(trust_env=False) as client:
            selected = credential_id(getattr(args, "credential_id", None))
            api = RadientTunnels(selected, client)
            result = (
                await api.request("GET", "/billing")
                if action == "billing"
                else await _ensure_billing(api, args.accept_monthly_price)
            )
        if action == "billing" and getattr(args, "json", False):
            # Emit only the billing contract, never arbitrary provider fields.
            # A successful request proves this selected login is valid now.
            return json.dumps(
                {
                    "credential_id": selected,
                    "account_valid": True,
                    "positive_balance": _positive_balance(result),
                    "setup_ready": _positive_balance(result) and result.get("eligible") is True,
                    "billing": {
                        key: result.get(key)
                        for key in (
                            "active",
                            "eligible",
                            "monthly_cost_usd",
                            "monthly_price_usd",
                            "balance_usd",
                            "amount_due_usd",
                        )
                    },
                },
                allow_nan=False,
            )
        return _billing_summary(result)
    if action == "list":
        async with httpx.AsyncClient(trust_env=False) as client:
            records = await RadientTunnels(
                credential_id(getattr(args, "credential_id", None)), client
            ).request("GET")
        if isinstance(records, dict):
            records = records.get("tunnels", [])
        return "\n\n".join(_summary(record) for record in records) or "No Radient tunnels."
    if action in {"create", "connect", "configure"}:
        old: dict[str, Any] = {}
        if (config.directory() / "config.json").exists():
            old = config.load()
        if action == "create" and old.get("tunnel_id"):
            raise ValueError(
                "A tunnel is already configured. Use configure or revoke before creating another."
            )
        selected = credential_id(getattr(args, "credential_id", None) or old.get("credential_id"))
        gateway_port = config.port(
            getattr(args, "gateway_port", None)
            or old.get("gateway_port", config.DEFAULT_GATEWAY_PORT)
        )
        local_auth = old.get("opencode_basic")
        if args.opencode_auth_file:
            local_auth = _read_origin_auth(args.opencode_auth_file)
        async with httpx.AsyncClient(trust_env=False) as client:
            api = RadientTunnels(selected, client)
            if action in {"create", "connect"}:
                # A lost create response or reconnect is recovery, not a new
                # enrollment. The cloud replays the same owner/key reservation
                # and still rejects a genuinely new reservation without credit.
                # Do not strand an existing connector at zero credit while its
                # established subscription remains above the -$1 floor.
                recovering = (
                    action == "create" and (config.directory() / "create.json").exists()
                ) or (
                    action == "connect"
                    and old.get("credential_id") == selected
                    and old.get("tunnel_id") is not None
                    and (args.tunnel_id or old["tunnel_id"]) == old["tunnel_id"]
                )
                await _ensure_billing(api, args.accept_monthly_price, setup=not recovering)
            if action == "create":
                harnesses = _harnesses(args)
                if any(h["port"] == gateway_port for h in harnesses):
                    raise ValueError("Gateway and harness ports must differ.")
                payload = {
                    "name": args.name or socket.gethostname(),
                    "device_id": old.get("device_id", str(uuid.uuid4())),
                    "gateway_port": gateway_port,
                    "harnesses": harnesses,
                }
                # Retain a create intent before contacting the cloud. If its
                # response is lost, rerunning the same command reuses the key
                # and cannot silently provision/bill a second connector.
                pending = config.directory() / "create.json"
                candidate = {
                    "credential_id": selected,
                    "key": str(uuid.uuid4()),
                    "payload": payload,
                }
                config.private_write(pending, json.dumps(candidate), exclusive=True)
                intent = json.loads(pending.read_text())
                if intent["credential_id"] != selected:
                    raise ValueError("A create request for another login is pending.")
                payload["device_id"] = intent["payload"]["device_id"]
                if payload != intent["payload"]:
                    raise ValueError(
                        "A different create request is pending; retry its original arguments."
                    )
                record = await api.request("POST", body=payload, idempotency_key=intent["key"])
            elif action == "connect":
                identifier = args.tunnel_id or old.get("tunnel_id")
                if not identifier:
                    raise ValueError("Supply the tunnel ID shown in the Radient console.")
                record = await api.request("GET", tunnel_path({"tunnel_id": identifier}))
                gateway_port = config.port(record["gateway_port"])
            else:
                if not old.get("tunnel_id"):
                    raise ValueError("Create or connect a tunnel first.")
                current = await api.request("GET", tunnel_path(old))
                if gateway_port != current["gateway_port"]:
                    # A remote ingress update reaches the existing cloudflared
                    # immediately, before a proof gateway can own the new port.
                    raise ValueError("Gateway port is fixed. Revoke and recreate to change it.")
                payload = {
                    "harnesses": _harnesses(args, current["harnesses"]),
                    "gateway_port": gateway_port,
                }
                if args.name:
                    payload["name"] = args.name
                if args.remote_enabled is not None:
                    if args.remote_enabled:
                        await _ensure_billing(api, args.accept_monthly_price)
                    payload["enabled"] = args.remote_enabled
                record = await api.request("PATCH", tunnel_path(old), body=payload)
        value = {
            "tunnel_id": record["id"],
            "credential_id": selected,
            "gateway_port": gateway_port,
            "device_id": record.get("device_id"),
            "record": record,
            "stopped": (
                bool(old.get("stopped"))
                if action == "configure"
                else action == "connect" and args.no_start
            ),
        }
        if old.get("tunnel_id") == record["id"] and old.get("credential_id") == selected:
            for key in ("cloudflared_path", "mobile_password"):
                if key in old:
                    value[key] = old[key]
        if local_auth is not None:
            value["opencode_basic"] = local_auth
        config.save(value)
        # Retain the intent until revocation. A concurrent create that began
        # before config.json was published must still reuse this reservation.
        if action == "connect" and not args.no_start:
            receipt = await dispatch(argparse.Namespace(tunnel_command="install"))
            return _summary(record) + "\n" + receipt
        return _summary(record) + "\nRun lop tunnel install to enable remote access on this device."
    value = config.load()
    if action == "status":
        record = value["record"]
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                record = await RadientTunnels(value["credential_id"], client).request(
                    "GET", tunnel_path(value)
                )
        except (ValueError, httpx.HTTPError):
            return _summary(record) + "\nCloud status unavailable; check /login radient."
        healthy = False
        connected = False
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                reply = await client.get(
                    f"http://127.0.0.1:{value['gateway_port']}/_lop_tunnel/health", timeout=2
                )
                healthy = reply.status_code == 200 and reply.json().get("ok") is True
                connected = healthy and reply.json().get("connected") is True
        except (httpx.HTTPError, ValueError):
            pass
        state = "connected" if connected else "connecting" if healthy else "stopped"
        return _summary(record) + f"\nLocal connector: {state}"
    if action == "stop":
        value["stopped"] = True
        config.save(value)
        try:
            install.action("stop")
        except ValueError:
            return (
                "Stop requested; foreground connector checks within 10 seconds. "
                "Local sessions continue."
            )
        return "Remote connector stopped; local operator sessions continue."
    if action == "revoke":
        value["stopped"] = True
        config.save(value)
        try:
            install.action("stop")
        except ValueError:
            pass
        async with httpx.AsyncClient(trust_env=False) as client:
            await RadientTunnels(value["credential_id"], client).request(
                "DELETE", tunnel_path(value)
            )
        install.uninstall()
        (config.directory() / "config.json").unlink(missing_ok=True)
        (config.directory() / "create.json").unlink(missing_ok=True)
        return "Tunnel revoked. Its public routes can no longer reach this device."
    if action == "uninstall":
        value["stopped"] = True
        config.save(value)
        install.uninstall()
        return "Local connector uninstalled. Use lop tunnel revoke to delete cloud routes too."
    if action in {"install", "start", "restart"}:
        value["cloudflared_path"] = await asyncio.to_thread(
            cloudflared_binary, value.get("cloudflared_path")
        )
        await asyncio.to_thread(_prepare_mobile, value)
        # User services do not inherit the terminal's environment. An explicit
        # foreground/Linux origin password stays in private local config;
        # Keychain-backed macOS installs need no redundant password copy.
        if os.environ.get("LOP_MOBILE_PASSWORD"):
            value["mobile_password"] = os.environ["LOP_MOBILE_PASSWORD"]
        value["stopped"] = False
        config.save(value)
        if action == "install":
            await asyncio.to_thread(install.install)
        else:
            await asyncio.to_thread(install.action, action)
        return (
            "Tunnel service started. Run lop tunnel status, then open your harness URL "
            "and log in with Radient."
        )
    raise ValueError(
        "Use lop tunnel --help to create, configure, start, stop, or revoke remote access."
    )


def main(args: argparse.Namespace) -> int:
    if args.tunnel_command == "serve":
        from local_operator.tunnels.service import main as serve

        try:
            value = config.load()
            value["stopped"] = False
            config.save(value)
        except (OSError, ValueError):
            print("Create or connect a tunnel before starting its connector.", file=sys.stderr)
            return 1
        return serve()
    try:
        print(asyncio.run(dispatch(args)))
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, httpx.HTTPError):
        print(
            "Tunnel operation failed; check network access and your Radient login.", file=sys.stderr
        )
        return 1


def mobile_action(action: str, accepted: str | None = None) -> str:
    """The TUI calls this in a worker thread; CLI and slash flows cannot drift."""
    from local_operator.tunnels.arguments import add_parser

    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())

    async def run() -> str:
        if action == "enable":
            receipts = []
            if not (config.directory() / "config.json").exists():
                args = ["tunnel", "create"]
                if accepted is not None:
                    args += ["--accept-monthly-price", accepted]
                receipts.append(await dispatch(parser.parse_args(args)))
            else:
                # An existing local config may refer to a remotely paused or
                # billing-suspended tunnel. Re-enable through the same explicit
                # quote acceptance before starting any connector service.
                args = ["tunnel", "configure", "--enable"]
                if accepted is not None:
                    args += ["--accept-monthly-price", accepted]
                receipts.append(await dispatch(parser.parse_args(args)))
            receipts.append(await dispatch(parser.parse_args(["tunnel", "install"])))
            return "\n".join(receipts)
        return await dispatch(parser.parse_args(["tunnel", action]))

    try:
        return asyncio.run(run())
    except ValueError as exc:
        return str(exc)
    except (OSError, httpx.HTTPError):
        return "Tunnel operation failed; check network access and /login radient."

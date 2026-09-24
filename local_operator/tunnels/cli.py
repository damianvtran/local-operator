"""Small CLI surface, also used by /mobile; no permanent agent-tool schema."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import socket
import sys
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

from local_operator import launchd
from local_operator.providers.auth_store import self_clearing_window
from local_operator.tunnels import config, gateway, report, state
from local_operator.tunnels.api import RadientTunnels, credential_id
from local_operator.tunnels.service import (
    authorization_failure_reason,
    cloudflared_binary,
    tunnel_path,
)


def _read_origin_auth(path: Path) -> dict[str, str]:
    """Read the OpenCode harness credentials, refusing a world-readable file.

    The mode check is a POSIX-ONLY heuristic, and gating it on ``os.name`` is
    the fix for a real defect rather than tidiness: on Windows every ordinary
    file reports ``st_mode & 0o077 == 0o066``, so this gate rejected THE CORRECT
    FILE every time — the OpenCode harness on a Radient tunnel was unusable, and
    the remedy the message named (``chmod 600``) does not exist on that
    platform. Windows protects the file with its profile ACL instead; the
    guarantee that still holds everywhere is the content check below, which runs
    on every read regardless of platform.
    """
    if os.name != "nt" and path.stat().st_mode & 0o077:
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


def _summary(record: dict[str, Any], source: str = "live", *, stopped: bool = False) -> str:
    """The cloud's own view of the tunnel, with its provenance on the status line.

    `source` is what stops the stale copy reading as live: when the cloud read
    fails, this command falls back to the record stored locally at the last
    `create`/`connect`/`configure`, and an operator reading `Status: active` off
    a cached copy — with the caveat trailing three lines below — is exactly how
    a withdrawn tunnel looked healthy.

    The explanatory line states PROVENANCE and no verdict (review round 1, D7).
    It used to print the relay vocabulary's own refusal sentence — "Radient
    refused the connector's authorization check, so the relay stopped serving" —
    immediately under a line that had just said the cloud could not be read: a
    verdict asserted by the command that could not obtain it, and one that
    repeated the sign-in advice and the billing link a third time. A local verdict
    belongs on the `Connector:` line, which reads this device; this line's job is
    only to say which copy of the cloud's record is being shown. `--json` still
    carries `cloud.reason` for a caller that wants the cause.

    `stopped` gates the line off entirely: nothing about a tunnel the operator
    deliberately stopped needs explaining, and the `(cached — cloud read failed)`
    marker on the line above already names the provenance (review round 1, D5).
    """
    provenance = "" if source == "live" else " (cached — cloud read failed)"
    lines = [
        f"Tunnel: {record.get('id', 'not created')}",
        f"Status: {record.get('status', 'configured')}{provenance}",
    ]
    if source != "live" and not stopped:
        lines.append("Cloud status: unavailable — showing the record stored at the last connect.")
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


def _stamp(seconds: Any) -> str:
    """A park's age on one line: the clock time today, the date too otherwise.

    A bare `14:32` on a park from last week would read as this afternoon, which
    is the one thing an operator must not get wrong about a state that has been
    holding since before they looked.
    """
    if not isinstance(seconds, int) or isinstance(seconds, bool):
        return ""
    when = datetime.datetime.fromtimestamp(seconds).astimezone()
    if when.date() == datetime.datetime.now().astimezone().date():
        return when.strftime("%H:%M")
    return when.strftime("%Y-%m-%d %H:%M")


def _status_text(
    payload: dict[str, Any], record: dict[str, Any], source: str, *, stopped: bool = False
) -> str:
    """`lop tunnel status`, human form: the connector leads, then the login.

    The connector first because that is the thing the operator is asking about,
    and because the old order put a cached cloud record — which can read
    `active` for a connector that is not running — at the top of the answer.

    ONE LINE, THEN A CONTINUATION (review round 1, D2). The first line used to be
    a single 317-cell paragraph: at 80 columns five rendered rows whose actionable
    clause sat on the third, with the state itself in the first three words, and
    the sign-in advice printed again on the `Login:` line and a third time in the
    cloud block. The read in a hurry is state → remedy → provenance, so the state
    line carries the state and its age, the park's own sentence follows indented
    beneath it, and the command appears once — on the `Login:` line, which is where
    this device's credential store reports its verdict.

    The command is appended HERE, from `TERMINAL_REMEDY`, and not baked into the
    sentence the park file carries: that sentence is also printed by the TUI card
    and forwarded to the desktop as `connector.detail`, so it names no command at
    all (review round 1, D1/M2) and each surface appends the one it can run.
    """
    connector = payload["connector"]
    line = f"Connector: {connector['state']}"
    if connector["reason"]:
        line += f" — {gateway.reason_label(connector['reason'])}"
    since = _stamp(connector.get("since"))
    if since:
        line += f" (since {since})"
    lines = [line]
    if connector["detail"]:
        # Indented, because the reason it is not appended to the line above is in
        # the docstring: a paragraph welded to the state is what buried the
        # command four rows down.
        lines.append(f"  {connector['detail']}")
    login = payload["login"]
    if login["state"] == "login_required":
        if stopped:
            # The fact stays, with its reason attached, and no command is offered:
            # remote access the operator switched off is not waiting on a sign-in,
            # which `report.remedy()` already honours for `--json` (review round
            # 1, D5).
            lines.append("Login: sign-in expired (not in use — tunnel stopped)")
        else:
            command = gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED]
            lines.append(f"Login: sign-in expired — run {command}")
    elif login["state"] == "deferred":
        # A state the store is WAITING OUT — the refresh token's last exchange is
        # unsettled — not a fault and not an expired login, so it names no command
        # (there is nothing to run: it clears by itself, and `report.remedy()`
        # returns `None` for it) and neither phrase above may appear here. This
        # case used to print "could not be checked", which was wrong twice: the
        # check did run, and no network had anything to do with it.
        #
        # The WINDOW is deliberately not on this line (UX round 1, U2): the
        # connector's own sentence, two rows up, carries it — that sentence is the
        # only one the desktop callout and the park file get (see
        # `TERMINAL_DETAIL[AUTHORIZATION_DEFERRED]`) — so repeating it here printed
        # the same fact twice and pushed the fact a reader needs onto the second
        # line of the screen. What stays here is the state, that it is retried
        # without anyone acting, and the escalation: the window re-arms on a
        # repeating stall (one deferral, not the state), so "nothing else is
        # needed" is a claim this line must not make.
        # The WINDOW and the escalation are stated once on the screen (UX round 1,
        # U2), and this line takes them only when nothing above already carries them:
        # the connector's own sentence — which travels to the desktop callout and the
        # park file, so it has to hold the number (see
        # `TERMINAL_DETAIL[AUTHORIZATION_DEFERRED]`) — is printed as the indented
        # detail whenever a gateway answered, and a second copy here printed one fact
        # twice and pushed it to the bottom. It is NOT printed for a STOPPED tunnel
        # (the state word says everything) or when no gateway was reachable, and there
        # this line is the only carrier left, so it names the window — derived from
        # the store's own bound rather than typed (agent review round 1, R2) — and the
        # escalation (UX round 1, U1: the window re-arms, so a deferral that keeps
        # returning is where a sign-in becomes the remedy).
        # The branch is decided by the REASON CODE, not by matching prose in a rendered
        # sentence (agent review round 2, R4). The payload carries the structural fact,
        # and it is exactly the condition under which the detail above IS this state's
        # own sentence: `report.probe` composes the detail from
        # `gateway.terminal_detail(reason, …)`, a parked tunnel carries the park's reason
        # and sentence together, and a stopped or unreachable one carries no reason and
        # no detail. Keying on the window's text instead let a copy edit that rewrote
        # the sentence AROUND the window silently flip which line a user sees, and flip
        # the duplication UX round 1 (U2) back in with it; the sentence still has to
        # carry the window, and that is pinned where the copy lives.
        #
        # The two branches exist because the screen varies: with a reason above, that row
        # already states the window and the escalation, so this line adds only the state;
        # with no reason above (a STOPPED tunnel, or a gateway that never answered) there
        # is no such row and this line is the only carrier, so it names the window —
        # derived from the store's own bound rather than typed (R2) — and the escalation
        # (U1: the window re-arms, so a deferral that keeps returning is where a sign-in
        # becomes the remedy).
        if connector["reason"] == gateway.AUTHORIZATION_DEFERRED:
            lines.append("Login: refresh deferred — retried automatically.")
        else:
            window = f"about {self_clearing_window()}"
            lines.append(
                f"Login: refresh deferred — it clears by itself within {window}; "
                "sign in again only if it persists."
            )
    elif login["state"] == "unknown":
        # Never "sign-in expired": this state means the check itself could not run,
        # and naming it anything else sends an operator whose network is down to a
        # login that cannot help them.
        lines.append("Login: could not be checked (a refresh could not reach Radient).")
    lines.extend(_summary(record, source, stopped=stopped).splitlines())
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
        # Self-heal first, silently: a parked connector plus a login that already
        # works (the operator re-authenticated before this build, or from another
        # surface) should not need a second command. This is the same call the
        # credential-write hook makes, so both paths agree on what re-arms.
        install.rearm_if_parked(provider="radient", credential_id=value["credential_id"])
        # The same read `report.local_payload` makes of the same key, because the
        # same hand is on the same file: a `config.json` that parses but carries
        # no `record` (hand-edited, or written by a build that stored it
        # elsewhere) is a configuration this command can still DESCRIBE — every
        # field `_summary` prints already defaults — not a bare
        # `KeyError: 'record'` rendered as a stack trace (review round 2, n2).
        stored = value.get("record")
        record = stored if isinstance(stored, dict) else {}
        source = "live"
        cloud_reason = ""
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                record = await RadientTunnels(value["credential_id"], client).request(
                    "GET", tunnel_path(value)
                )
        except (ValueError, httpx.HTTPError) as failure:
            # A network fault and an unusable login are different jobs for the
            # operator, and one shared line sent both to /login radient. Telling
            # them apart takes the classifier, not the exception class: a refresh
            # that could not reach Radient during this very request arrives as a
            # ValueError too (see `RadientTunnels.request`), and the cloud read is
            # unavailable in both cases, so this is the only surface that can.
            #
            # The read failing is not fatal to the command any more: the cached
            # record is still printed, now clearly marked as cached, and the
            # connector's own state is read from this device where the answer
            # actually is.
            source = "cached"
            cloud_reason = authorization_failure_reason(failure)
        connector = await report.connector_state(value)
        login = await report.login_verdict(value)
        payload = report.payload(
            value,
            record,
            source=source,
            cloud_reason=cloud_reason,
            connector=connector,
            login=login,
        )
        if getattr(args, "json", False):
            return json.dumps(payload, allow_nan=False)
        return _status_text(payload, record, source, stopped=bool(value.get("stopped")))
    if action == "stop":
        value["stopped"] = True
        config.save(value)
        # A deliberate stop is not a park: the operator is not using the tunnel,
        # so every surface must stop describing a connector that is waiting on
        # them. Cleared here as well as in `run` because `bootout` can take the
        # process out without it ever reaching its own stopped branch.
        state.clear()
        try:
            install.action("stop")
        except (launchd.JobNotOurs, launchd.IdentityUnverifiable):
            # A REFUSAL IS NOT "NO SUPERVISED JOB" (review round 3, QA Q-2). The
            # branch below is right for the foreground case and was a false
            # success for this one: the identity guard declined, nothing was
            # called, and this printed "Stop requested …" with exit 0. Letting it
            # reach the outer handler prints the refusal's own sentence and exits
            # non-zero, which is what the operator needs to know. Both refusal
            # types, because "cannot tell" is a refusal too (round 4).
            raise
        except ValueError:
            return (
                "Stop requested; foreground connector checks within 10 seconds. "
                "Local sessions continue."
            )
        return "Remote connector stopped; local operator sessions continue."
    if action == "revoke":
        value["stopped"] = True
        config.save(value)
        state.clear()
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
        state.clear()
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
        # `lop login radient`: every other line this command prints names the
        # remedy this shell can run, and this one used to be the exception
        # (review round 2, m5).
        return "Tunnel operation failed; check network access and lop login radient."

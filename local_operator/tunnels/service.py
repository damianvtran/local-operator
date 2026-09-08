"""Supervise the loopback gateway and one cloudflared connector as one unit.

A terminal session never owns their lifetime. launchd/systemd supervises this
process; stopping it withdraws remote access without terminating agent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import socket
import subprocess
from typing import Any
from urllib.parse import quote

import httpx

from local_operator.mobile.auth import load_password
from local_operator.tunnels import config
from local_operator.tunnels.api import RadientTunnels
from local_operator.tunnels.gateway import Gateway

POLL_SECONDS = 10


def cloudflared_binary(configured: str | None = None) -> str:
    binary = configured or shutil.which("cloudflared")
    if not binary:
        raise ValueError("Install cloudflared 2025.4.0 or newer, then run lop tunnel install.")
    result = subprocess.run(
        [binary, "tunnel", "run", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode or "--token-file" not in result.stdout:
        raise ValueError("Update cloudflared to 2025.4.0 or newer (token-file support required).")
    return binary


def tunnel_path(value: dict[str, Any]) -> str:
    identifier = value.get("tunnel_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Missing configured tunnel ID.")
    return "/" + quote(identifier, safe="")


def pinned_harness_ports(value: dict[str, Any]) -> tuple[dict[str, int] | None, str]:
    """The harness ports the operator last approved from this device.

    `lop tunnel create/connect/configure` persist the whole cloud record
    locally, so `record.harnesses` is the device's own copy of the port map an
    operator ran a local command to accept. Returns `(None, reason)` — meaning
    "no usable pin" — for a record written before the field existed or
    hand-edited into a shape we cannot trust, so callers can fall back rather
    than strand a device whose config predates this check.

    One unusable entry disables the pin for the whole record, so the reason
    names that entry: without it an operator reading the fallback warning knows
    the pin is off but not which line of config.json to repair.
    """
    record = value.get("record")
    harnesses = record.get("harnesses") if isinstance(record, dict) else None
    if not isinstance(harnesses, list) or not harnesses:
        return None, "this tunnel configuration predates it"
    pinned: dict[str, int] = {}
    for position, harness in enumerate(harnesses, start=1):
        if not isinstance(harness, dict) or not isinstance(harness.get("id"), str):
            return None, f"stored harness entry {position} has no usable id"
        if harness["id"] in pinned:
            # The same silent last-wins ambiguity the JWKS loader refuses on a
            # repeated kid (gateway.OriginVerifier): two entries claiming one
            # id leave the approved port decided by serialization order rather
            # than by the operator, and this file is hand-editable. Fall back
            # and name it instead of picking a winner. Falling back rather than
            # raising keeps the untrustworthy-shape contract above: a bad local
            # record must never strand a device that the cloud still serves.
            return None, f"stored harness {harness['id']} is listed twice"
        try:
            pinned[harness["id"]] = config.port(harness.get("port"))
        except ValueError:
            return None, f"stored harness {harness['id']} has an unusable port"
    return pinned, ""


def enforce_harness_ports(connection: dict[str, Any], value: dict[str, Any]) -> None:
    """Refuse a /connect that repoints a harness at an unapproved local port.

    The cloud validates a harness port only as 1024-65535 and != gateway_port,
    and replaces the harness list wholesale on PATCH, so anyone holding the
    owner's console session can aim a harness at any other loopback service.
    The gateway then attaches this device's mobile-relay cookie or OpenCode
    Basic credential to whatever answers there (gateway.headers), handing a
    live relay credential to an unrelated local service. Pinning against the
    locally stored record makes changing a harness port require an act on this
    machine, exactly as the cloud already treats gateway_port.

    Fail-closed but recoverable: the console PATCH that legitimately changes a
    port bumps the tunnel version, the poller restarts, and this refusal then
    names the local command that re-approves it. Refreshing the pin is left to
    those commands (they are what write `record`); re-pinning from the polled
    record here would restore exactly the silent repointing this prevents.
    """
    pinned, unpinnable = pinned_harness_ports(value)
    if pinned is None:
        print(
            f"Warning: harness port pinning is inactive ({unpinnable}), so console port "
            "changes are not verified locally. Run lop tunnel connect to pin them.",
            flush=True,
        )
        return
    for harness in connection["tunnel"]["harnesses"]:
        # A disabled harness is never dialed (Gateway.harness filters on it),
        # so its port cannot carry a credential and is not worth an outage.
        if not harness["enabled"]:
            continue
        approved = pinned.get(harness["id"])
        if approved is None:
            # A harness the console ADDED since the last local command. The
            # remedy is the same, but reporting it as a port change describes
            # an event that did not happen and sends the operator hunting for
            # a port they never set.
            raise ValueError(
                f"Harness {harness['id']} is not approved on this device. "
                "Run lop tunnel connect again."
            )
        if approved != harness["port"]:
            raise ValueError(
                f"Harness port for {harness['id']} changed in the console. "
                "Run lop tunnel connect again."
            )


def active(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("enabled") is True
        and record.get("status") == "active"
        and (
            "billing" not in record
            or (isinstance(record["billing"], dict) and record["billing"].get("eligible") is True)
        )
    )


async def run() -> int:
    import uvicorn

    value = config.load()
    if value.get("stopped"):
        # A user-level service may run again at the next login. A persisted
        # explicit stop is successful, so supervisors must not keep retrying it.
        return 0
    binary = cloudflared_binary(value.get("cloudflared_path"))
    # Bind before contacting the cloud or starting cloudflared: a conflicting
    # listener must never receive an enabled public route even momentarily.
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", config.port(value["gateway_port"])))
        listener.listen(128)
        listener.setblocking(False)
    except BaseException:
        listener.close()
        raise
    token_file = config.directory() / "cloudflared.token"
    ready_file = config.directory() / "cloudflared.pid"
    child: asyncio.subprocess.Process | None = None
    scanner: asyncio.Task[None] | None = None
    server: Any = None
    serve_task: asyncio.Task[Any] | None = None
    gateway: Gateway | None = None
    restart = False
    try:
        # Proxy configuration and environment credentials are not forwarded to
        # either Radient or loopback harnesses. Both use explicit destinations.
        async with httpx.AsyncClient(
            trust_env=False, timeout=httpx.Timeout(15, read=None)
        ) as client:
            api = RadientTunnels(int(value["credential_id"]), client)
            connection = config.validate_connection(
                await api.request("POST", tunnel_path(value) + "/connect")
            )
            if not active(connection["tunnel"]):
                raise ValueError(
                    "Tunnel disabled or billing suspended. Review it in the Radient console."
                )
            if connection["gateway_port"] != value["gateway_port"]:
                raise ValueError(
                    "Gateway port changed in the console. Run lop tunnel connect again."
                )
            enforce_harness_ports(connection, value)
            needs_mobile = any(
                h["enabled"] and h["id"] == "local-operator"
                for h in connection["tunnel"]["harnesses"]
            )
            password = await asyncio.to_thread(load_password) if needs_mobile else None
            if needs_mobile and password is None:
                password = value.get("mobile_password")
            if needs_mobile and not password:
                raise ValueError("Install the mobile relay with lop mobile install first.")
            gateway = Gateway(
                connection,
                client,
                mobile_password=password,
                opencode_basic=value.get("opencode_basic"),
                connector_ready=lambda: (
                    ready_file.exists() and child is not None and child.returncode is None
                ),
            )
            token = connection.get("cloudflared_token")
            if not isinstance(token, str) or not token or "\n" in token:
                raise ValueError("Radient did not return a connector token.")
            config.private_write(token_file, token)
            ready_file.unlink(missing_ok=True)
            # No credentials in argv, inherited environment, or logs. A
            # per-tunnel token authorizes only its connector, never Cloudflare
            # account administration. cloudflared debug logging is forbidden.
            child_env = {
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "HOME", "SYSTEMROOT", "SSL_CERT_FILE", "SSL_CERT_DIR"}
            }
            child = await asyncio.create_subprocess_exec(
                binary,
                "tunnel",
                "--no-autoupdate",
                "--loglevel",
                "error",
                "--grace-period",
                "1s",
                "--pidfile",
                str(ready_file),
                "run",
                "--token-file",
                str(token_file),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            server = uvicorn.Server(
                uvicorn.Config(
                    gateway.app(),
                    host="127.0.0.1",
                    port=value["gateway_port"],
                    log_level="warning",
                    access_log=False,
                    proxy_headers=False,
                    timeout_graceful_shutdown=2,
                )
            )
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except NotImplementedError:
                    # Windows event loops do not implement add_signal_handler.
                    # The synchronous handler only schedules work on the loop.
                    signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))

            async def poll() -> None:
                nonlocal restart
                assert gateway is not None
                while not stop.is_set():
                    try:
                        if config.load().get("stopped"):
                            gateway.revoked = True
                            stop.set()
                            return
                        record = await api.request("GET", tunnel_path(value))
                        # Configuration changes invalidate all signed requests
                        # from the old version. Restart for a new proof context,
                        # without ever guessing how to merge trust boundaries.
                        if (
                            not active(record)
                            or record.get("version") != connection["tunnel"]["version"]
                        ):
                            # Resume after console reactivation or a credit
                            # top-up. User services retry at their 10s floor;
                            # /connect refuses publication while suspended.
                            restart = True
                            gateway.revoked = True
                            stop.set()
                            return
                        gateway.authorize()
                    except (ValueError, httpx.HTTPError):
                        # The gateway's short authorization lease closes even
                        # when the control-plane network is unavailable.
                        pass
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
                    except TimeoutError:
                        pass

            scanner = asyncio.create_task(poll())
            serve_task = asyncio.create_task(server.serve(sockets=[listener]))
            stopping = asyncio.create_task(stop.wait())
            exited = asyncio.create_task(child.wait())
            try:
                await asyncio.wait(
                    {serve_task, stopping, exited}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                stopping.cancel()
                exited.cancel()
                await asyncio.gather(stopping, exited, return_exceptions=True)
            # Crashes and remote suspension are restartable. Only an explicit
            # local stop exits successfully and stays stopped under supervision.
            return 1 if restart or child.returncode is not None or serve_task.done() else 0
    finally:
        if gateway is not None:
            gateway.revoked = True
        if scanner is not None:
            scanner.cancel()
            await asyncio.gather(scanner, return_exceptions=True)
        if server is not None:
            server.should_exit = True
        if serve_task is not None:
            await serve_task
        if child is not None and child.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                child.terminate()
            try:
                await asyncio.wait_for(child.wait(), timeout=5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    child.kill()
                await child.wait()
        listener.close()
        token_file.unlink(missing_ok=True)
        ready_file.unlink(missing_ok=True)


def main() -> int:
    try:
        return asyncio.run(run())
    except ValueError as failure:
        # Service logs contain operational state, never upstream bodies,
        # request URLs, or a traceback containing credential arguments, and
        # printing a ValueError's text keeps that property: every ValueError
        # this package raises is a fixed literal carrying at most a harness id,
        # a tunnel status, or an HTTP status code (api.py deliberately refuses
        # to echo provider bodies). The only ValueErrors run() does not author
        # are json.JSONDecodeError on a corrupt config.json or /connect body,
        # whose message is a byte offset and never the document, and int() on a
        # hand-edited credential_id, which is a local row id and not a secret.
        # This is the only place the remedy is written down: suppressing it is
        # what left the harness-port refusal telling the operator to check a
        # Radient login that is fine, with nothing naming lop tunnel connect.
        print(f"Tunnel connector stopped: {failure}", flush=True)
        return 1
    except (OSError, httpx.HTTPError):
        # These carry text this module did not author: httpx echoes the full
        # request URL (query string included) and OSError echoes filesystem
        # paths, so only the generic line is safe here.
        print(
            "Tunnel connector stopped; check lop tunnel status and your Radient login.", flush=True
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

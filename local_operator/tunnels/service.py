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
    except (ValueError, OSError, httpx.HTTPError):
        # Service logs contain operational state, never upstream bodies,
        # request URLs, or a traceback containing credential arguments.
        print(
            "Tunnel connector stopped; check lop tunnel status and your Radient login.", flush=True
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

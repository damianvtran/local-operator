"""Run the mobile daemon: ``lop mobile serve`` lands here.

Foreground by design — supervision belongs to launchd (the LaunchAgent this
repo's installer writes) or a developer's terminal, never to a self-daemonizing
double-fork. The runner resolves the password, binds loopback, starts the
record scanner, and serves until SIGTERM.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from local_operator.mobile.auth import load_password
from local_operator.mobile.daemon import DEFAULT_PORT, MobileDaemon, build_app

logger = logging.getLogger(__name__)


async def amain(port: int = DEFAULT_PORT) -> int:
    password = load_password()
    if not password:
        # First-run is an operator action, not a silent default: the daemon
        # refuses to bind unauthenticated rather than guess a password.
        print(
            "no mobile password set. Run `lop mobile install` (or set "
            "LOP_MOBILE_PASSWORD) first.",
            flush=True,
        )
        return 2

    daemon = MobileDaemon(port=port, password=password)
    app = build_app(daemon)

    import uvicorn

    config = uvicorn.Config(
        app,
        host="127.0.0.1",  # THE security invariant: loopback only, always.
        port=port,
        log_level="warning",
        # SSE holds connections open for the phone's whole session; uvicorn's
        # default keepalive timeout would reap them between events.
        timeout_keep_alive=75,
    )
    server = uvicorn.Server(config)

    scanner = asyncio.ensure_future(daemon.scan_loop())

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    serve_task = asyncio.ensure_future(server.serve())
    stop_task = asyncio.ensure_future(stop.wait())
    done, _pending = await asyncio.wait(
        {serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
    )
    if stop_task in done:
        server.should_exit = True
        await serve_task
    scanner.cancel()
    return 0


def main(port: int = DEFAULT_PORT) -> int:
    try:
        return asyncio.run(amain(port))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    # ``python -m local_operator.mobile.service`` is what the LaunchAgent
    # runs: re-entering the installed package means an upgrade changes what
    # the supervised process runs with no reinstall step.
    import argparse

    parser = argparse.ArgumentParser(prog="lop mobile serve")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    raise SystemExit(main(parser.parse_args().port))

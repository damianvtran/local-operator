"""Run the mobile daemon: ``lop mobile serve`` lands here.

Foreground by design — supervision belongs to launchd (the LaunchAgent this
repo's installer writes) or a developer's terminal, never to a self-daemonizing
double-fork. The runner resolves the password, binds loopback, starts the
record scanner, and serves until SIGTERM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from local_operator import supervisors
from local_operator.logger import configure_console_logging, quiet_wire_clients
from local_operator.mobile.auth import load_password, store_description
from local_operator.mobile.daemon import DEFAULT_PORT, MobileDaemon, build_app
from local_operator.procstate import install_loop_signal_handlers

logger = logging.getLogger(__name__)


async def amain(port: int = DEFAULT_PORT) -> int:
    # The daemon's stderr IS its log file: the LaunchAgent this repo installs
    # points StandardOutPath and StandardErrorPath at log_dir()/mobile.log. So
    # this process has to own that stream, because unconfigured it inherits
    # whatever a dependency's `basicConfig` installed — the MCP client's is
    # `level=INFO` — and one record per HTTP request is what filled the relay
    # log with hundreds of thousands of unusable lines.
    #
    # The wire-client pin is called explicitly rather than left to
    # `configure_console_logging`: that function quietens those libraries only as
    # a side effect of the level it is given, so a future call here with
    # `level=INFO` (a reasonable-looking change, since a daemon's own records are
    # what a reader wants) would silently restore the flood.
    configure_console_logging()
    quiet_wire_clients()
    password = load_password()
    if not password:
        # First-run is an operator action, not a silent default: the daemon
        # refuses to bind unauthenticated rather than guess a password.
        #
        # WHICH COMMAND THIS NAMES DEPENDS ON THE HOST (audit A23). `lop mobile
        # install` supervises the daemon through launchd, systemd --user or Task
        # Scheduler, and a host that has none of those — a container, OpenRC/
        # Alpine, a Linux without systemd — refuses it outright
        # (``supervisors.no_supervisor_error``). Telling that operator to run it
        # names a command that cannot work, so this branches on the same
        # discovery the installer uses. ``store_description()`` is the other
        # half: the answer used to be "the Keychain" on every platform, which
        # is one of the three stores this now picks between.
        if supervisors.supervisor() is None:
            print(
                "no mobile password set, and this host has no user service "
                "supervisor, so `lop mobile install` cannot run here. Run "
                f"`lop mobile password` to store one in {store_description()}, "
                "then run the daemon in the foreground with `lop mobile serve` "
                "(or set LOP_MOBILE_PASSWORD).",
                flush=True,
            )
        else:
            print(
                "no mobile password set. Run `lop mobile install` (it stores "
                f"the password in {store_description()}), or set "
                "LOP_MOBILE_PASSWORD first.",
                flush=True,
            )
        return 2

    # A SECOND daemon on the same machine must never dial registrants: each
    # admits at most one daemon connection, so a secondary dial would evict
    # the production daemon's live bridge mid-session. LO_MOBILE_NO_DIAL=1
    # runs this instance as a read-only observer (list + durable routes only)
    # — the mode benchmark and diagnostic daemons use.
    dial = os.environ.get("LO_MOBILE_NO_DIAL", "").strip().lower() not in ("1", "true", "yes")
    daemon = MobileDaemon(port=port, password=password, dial_registrants=dial)
    if not dial:
        logger.warning("mobile daemon running in no-dial observer mode")
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
    # `loop.add_signal_handler` is Unix-only, and the Windows Proactor loop's
    # inherited stub raises NotImplementedError — which escaped `amain` and
    # killed the relay at startup, before it could bind. The helper falls back
    # to `signal.signal` there. `lop mobile serve` is the ONLY supported way to
    # run this daemon off macOS (`lop mobile install` refuses), so this is the
    # difference between the relay working on Windows and not existing.
    install_loop_signal_handlers(loop, {signal.SIGTERM: stop.set, signal.SIGINT: stop.set})

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
    # Linux comm axis: the LaunchAgent/unit names the IMAGE on macOS and this
    # daemon's `comm` has to be set by the process itself there (see
    # :func:`procname.brand_this_process`; a no-op on macOS).
    from local_operator import procname

    procname.brand_this_process()
    # ``python -m local_operator.mobile.service`` is what the LaunchAgent
    # runs: re-entering the installed package means an upgrade changes what
    # the supervised process runs with no reinstall step.
    import argparse

    parser = argparse.ArgumentParser(prog="lop mobile serve")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    raise SystemExit(main(parser.parse_args().port))

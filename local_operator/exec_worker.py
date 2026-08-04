"""Detached background exec worker — ``python -m local_operator.exec_worker``.

This is the process ``exec --background`` spawns (see ``exec_mode``): it
receives the prompt and selectors via argv, builds a session through the
shared composition root, runs exactly one prompt headless, and exits with
the run's outcome. The parent CLI never waits on it (``start_new_session``);
stdout/stderr are already redirected to the job log by the spawner.

SIGTERM safety: SIGTERM is the expected shutdown signal for these detached
jobs. The handler aborts the running turn and lets the async main flush its
renderer output and dispose the session before exiting 130 — a hard kill
here would truncate the log mid-write and leak provider connections.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import Any, Callable

#: Exit code for an interrupted (SIGTERM/SIGINT) run — distinct from the
#: engine's 0/1 success/error codes so job ledgers can tell them apart.
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """Parse the worker's mirror flags (see ``exec_mode.build_worker_argv``)."""
    parser = argparse.ArgumentParser(
        prog="local_operator.exec_worker",
        description="Run one local-operator prompt headless (background exec worker)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to execute")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="Emit one JSON line per agent event",
    )
    parser.add_argument("--yolo", action="store_true", help="Auto-approve all tool tiers")
    parser.add_argument("--agent", type=str, default=None, help="Agent name selector")
    parser.add_argument("--agent-id", type=str, default=None, dest="agent_id", help="Agent id")
    parser.add_argument("--hosting", type=str, default=None, help="Hosting platform override")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    return parser


def _install_sigterm_handler(loop: asyncio.AbstractEventLoop, session_box: list) -> None:
    """SIGTERM -> abort the running turn; the run loop then disposes cleanly.

    ``session_box`` holds the live session once constructed (a list because
    the handler is installed before the session exists). Best-effort: a
    platform without loop signal support falls back to default handling.
    """

    def handler() -> None:
        session = session_box[0] if session_box else None
        if session is not None:
            try:
                session.abort("terminated")
            except Exception:  # noqa: BLE001 — must never raise in a handler
                pass
        # Give the turn a bounded window to settle and flush; then hard-exit
        # the loop. The finally block in ``run`` disposes the session.
        loop.call_later(5.0, loop.stop)

    try:
        loop.add_signal_handler(signal.SIGTERM, handler)
    except (NotImplementedError, RuntimeError):
        # Non-POSIX or pre-loop environments: fall back to a plain handler
        # that cannot schedule into the loop.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(EXIT_INTERRUPTED))


def _default_session_factory(parsed: argparse.Namespace):
    """Build the real session via the shared composition root.

    Returns an awaitable session; all engine imports stay lazy inside
    ``session_factory.create_session``.
    """
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import create_session

    session_args = argparse.Namespace(
        hosting=parsed.hosting,
        model=parsed.model,
        agent_name=parsed.agent,
        agent_id=parsed.agent_id,
        yolo=parsed.yolo,
        train=False,
    )
    config_dir = Path.home() / ".local-operator"
    config_manager = ConfigManager(config_dir)
    credential_manager = CredentialManager(config_dir)

    from local_operator.agents import AgentRegistry  # lazy: heavy module

    agent_registry = AgentRegistry(config_dir)
    return create_session(session_args, config_manager, credential_manager, agent_registry)


def run(parsed: argparse.Namespace, session_factory: "Callable[[], Any] | None" = None) -> int:
    """Build the session, run one prompt, return the exit code.

    The engine's ``prompt`` never raises on provider errors (stream A
    contract — errors surface as ``agent_end`` with ``error`` set), so the
    exit code comes straight from ``run_print_mode``'s renderer tracking.

    ``session_factory`` is injectable for tests; the default wires the real
    engine through the shared composition root.
    """
    from local_operator.headless_print import run_print_mode

    factory = session_factory or (lambda: _default_session_factory(parsed))

    async def async_main() -> int:
        loop = asyncio.get_running_loop()
        session_box: list = []
        _install_sigterm_handler(loop, session_box)

        session = factory()
        if asyncio.iscoroutine(session):
            session = await session
        session_box.append(session)
        try:
            return await run_print_mode(session, [parsed.prompt], json_mode=parsed.json_mode)
        except asyncio.CancelledError:
            return EXIT_INTERRUPTED

    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


def main() -> int:
    """Console entry: parse argv, run, flush, exit."""
    parsed = build_parser().parse_args()
    try:
        code = run(parsed)
    except Exception as exc:  # noqa: BLE001 — a log file is the only surface
        sys.stderr.write(f"exec_worker error: {exc}\n")
        code = 1
    # Flush before returning: the spawner owns this file's lifetime and the
    # process may be reaped right after exit.
    sys.stdout.flush()
    sys.stderr.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())

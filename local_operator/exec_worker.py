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
import inspect
import logging
import os
import signal
import sys
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    # Type-only: the worker's whole design is that engine imports stay lazy
    # so a background spawn pays for them once, inside the factory.
    from local_operator.session.protocol import SessionProtocol

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
    parser.add_argument(
        "--train",
        action="store_true",
        help="Training mode: append the transcript to the agent directory (legacy --train)",
    )
    parser.add_argument("--agent", type=str, default=None, help="Agent name selector")
    parser.add_argument("--agent-id", type=str, default=None, dest="agent_id", help="Agent id")
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        dest="job_id",
        help="Ledger job id; when set, the worker appends a terminal record"
        " (finished_at + exit_code) to the JSONL ledger on exit",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Publish a session record and serve the control socket for this run",
    )
    parser.add_argument("--hosting", type=str, default=None, help="Hosting platform override")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="SESSION_ID",
        # A background job is the same request run elsewhere, so it has to be able
        # to continue a session. Without this the flag was accepted by the front
        # end, dropped at the process boundary, and the worker started a fresh
        # session while reporting success.
        help="Resume a previous session by id (or '@latest')",
    )
    from local_operator.exec_startup import add_startup_arguments

    add_startup_arguments(parser)
    return parser


def _install_sigterm_handler(
    loop: asyncio.AbstractEventLoop,
    session_box: list[SessionProtocol],
    interrupted: asyncio.Event,
) -> None:
    """SIGTERM -> signal ``interrupted``; async_main returns 130 (CL-03).

    ``session_box`` holds the live session once constructed (a list because
    the handler is installed before the session exists). The handler aborts
    the running turn (best effort) and sets the event; ``async_main`` races
    the turn against the event and owns the exit code — the handler never
    stops the loop itself, so disposal and flushing stay in normal flow.
    Best-effort: a platform without loop signal support falls back to
    default handling.
    """

    def handler() -> None:
        session = session_box[0] if session_box else None
        if session is not None:
            try:
                session.abort("terminated")
            except Exception:  # noqa: BLE001 — must never raise in a handler
                pass
        interrupted.set()

    try:
        loop.add_signal_handler(signal.SIGTERM, handler)
    except (NotImplementedError, RuntimeError):
        # Non-POSIX or pre-loop environments: fall back to a plain handler
        # that cannot schedule into the loop.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(EXIT_INTERRUPTED))


def _install_sighup_ignore(loop: asyncio.AbstractEventLoop) -> None:
    """SIGHUP is IGNORED: this worker's lifetime is not an interface's to end.

    A background worker is spawned detached (``start_new_session=True``) and
    writes its log to a file, so losing a controlling terminal is not a reason
    to drop a run half-done. Left at the default disposition a HUP kills the
    interpreter outright, which truncates the job log mid-write, skips the
    ledger's terminal record (``--job-id``) and leaks provider connections —
    the same hard-exit shape SIGTERM handling exists to prevent.

    SIGTERM remains the one signal that ends this run (``_install_sigterm_handler``);
    a HUP only produces a bounded, one-shot log line, which is why ``main()``
    configures console logging: the job log IS this process's stderr, and a
    survival nobody can find in the record is the attribution gap this PR exists
    to close (review round 1, MINOR-1).
    """
    hup_logged = False

    def handler() -> None:
        nonlocal hup_logged
        if hup_logged:
            return
        hup_logged = True
        logging.getLogger(__name__).info(
            "exec worker: ignoring SIGHUP (pid %d); this worker is detached from interfaces",
            os.getpid(),
        )

    # ``SIGHUP`` is POSIX-only; a platform without it must not fail to start a
    # worker over a signal it could not have received. The same shape as
    # ``session/runtime/process.amain``, which ignores a HUP for the same
    # reason.
    sighup = getattr(signal, "SIGHUP", None)
    if sighup is None:
        return
    try:
        loop.add_signal_handler(sighup, handler)
        return
    except (NotImplementedError, RuntimeError, ValueError):
        pass
    # THE FALLBACK CANNOT BE ALLOWED TO RAISE, and it is the branch that plants
    # an INHERITABLE ignore: ``signal.signal`` works only on the main thread
    # (``ValueError`` otherwise, which is how a loop that refused for that reason
    # would then kill the run this function protects), and a SIG_IGN survives
    # ``exec`` — CPython's ``restore_signals`` resets only SIGPIPE/SIGXFZ/SIGXFSZ
    # — so anything spawned after it would inherit an ignored HUP. Nothing
    # reaches here on the shipped path (the loop takes the callback).
    try:
        signal.signal(sighup, signal.SIG_IGN)
    except (ValueError, OSError, RuntimeError):
        logging.getLogger(__name__).warning("could not install the SIGHUP ignore", exc_info=True)


def _default_session_factory(parsed: argparse.Namespace) -> Awaitable[SessionProtocol]:
    """Build the real session via the shared composition root.

    Returns an awaitable session; all engine imports stay lazy inside
    ``session_factory.create_session``.
    """
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.paths import config_dir
    from local_operator.session_factory import create_session

    session_args = argparse.Namespace(
        hosting=parsed.hosting,
        model=parsed.model,
        agent_name=parsed.agent,
        agent_id=parsed.agent_id,
        yolo=parsed.yolo,
        train=bool(getattr(parsed, "train", False)),
        resume=parsed.resume,
    )
    # config_dir(), not ``Path.home() / ".local-operator"``: a missed copy of the
    # hardcoded root that ``exec_mode._make_default_session_factory`` already
    # fixed for the foreground path (see the comment there). This is the
    # BACKGROUND worker, which inherits the spawner's environment, so leaving it
    # hardcoded made ``exec --background`` ignore LOCAL_OPERATOR_CONFIG_DIR while
    # the foreground run honoured it — the same entry point resolving two
    # different roots depending on a flag. It also reaches the analytics
    # session-name backfill through ``create_session``'s store-maintenance pass,
    # which writes to whatever root it is handed.
    base_dir = config_dir()
    config_manager = ConfigManager(base_dir)
    credential_manager = CredentialManager.readonly(base_dir)

    from local_operator.agents import AgentRegistry  # lazy: heavy module

    agent_registry = AgentRegistry(base_dir)
    return create_session(session_args, config_manager, credential_manager, agent_registry)


def run(
    parsed: argparse.Namespace,
    session_factory: Callable[[], SessionProtocol | Awaitable[SessionProtocol]] | None = None,
) -> int:
    """Build the session, run one prompt, return the exit code.

    The engine's ``prompt`` never raises on provider errors (stream A
    contract — errors surface as ``agent_end`` with ``error`` set), so the
    exit code comes straight from ``run_print_mode``'s renderer tracking.

    ``session_factory`` is injectable for tests; the default wires the real
    engine through the shared composition root.
    """
    from local_operator.exec_session import run_session
    from local_operator.exec_startup import resolve_startup

    # Worker preflight precedes even the injected session factory. The legacy
    # selector is called agent_name by the launcher and agent on worker argv.
    parsed.agent_name = parsed.agent
    team = resolve_startup(parsed)
    factory = session_factory or (lambda: _default_session_factory(parsed))

    async def async_main() -> int:
        loop = asyncio.get_running_loop()
        interrupted = asyncio.Event()
        session_box: list[SessionProtocol] = []
        _install_sigterm_handler(loop, session_box, interrupted)
        # After the kill switch, deliberately: the two handlers are independent,
        # but a failure inside this registration (``signal.signal`` on a
        # non-main thread raises) must not be able to cost the run its SIGTERM
        # path. Ordering the terminating disposition first makes that failure
        # harmless instead of fatal.
        _install_sighup_ignore(loop)

        async def execute() -> int:
            source = factory()
            session: SessionProtocol = await source if inspect.isawaitable(source) else source
            session_box.append(session)
            return await run_session(session, parsed.prompt, parsed, team)

        # Race the WHOLE lifetime, not just the first prompt: a worker waiting
        # on provider discovery must still honour termination and finalize.
        prompt_task = asyncio.ensure_future(execute())
        interrupt_task = asyncio.ensure_future(interrupted.wait())
        await asyncio.wait({prompt_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED)
        if interrupted.is_set():
            # Give the turn a bounded window to settle (flush renderer output,
            # dispose the session) before reporting the interrupt.
            prompt_task.cancel()
            try:
                await asyncio.wait_for(prompt_task, timeout=5.0)
            except (Exception, asyncio.CancelledError):
                # The explicit interrupt owns the outcome, including when the
                # provider was initializing rather than streaming a turn.
                pass
            return EXIT_INTERRUPTED
        interrupt_task.cancel()
        try:
            return prompt_task.result()
        except asyncio.CancelledError:
            # A supervisor `stop` cancels the whole lifetime rather than
            # signalling the process, and ``CancelledError`` is a
            # BaseException — unhandled it would skip main()'s terminal ledger
            # write, leaving reconciliation to report a deliberate stop as an
            # abrupt `interrupted`. Deliberate termination is `cancelled`.
            return EXIT_INTERRUPTED

    try:
        return asyncio.run(async_main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        return EXIT_INTERRUPTED


def main() -> int:
    """Console entry: parse argv, run, flush, exit.

    When the spawner passed ``--job-id`` (CL-09), the worker appends the
    terminal ledger record (``finished_at`` + ``exit_code``) before exiting —
    best effort; ledger bookkeeping must never change the exit code.

    The SIGHUP ignore is armed on the first lines of ``async_main`` — as early
    as the loop allows, so it covers the turn and everything the turn builds. The
    residual window is this function's argv/preflight work and the loop
    bootstrap, and it is deliberately not closed with a process-wide
    ``SIG_IGN`` set here: this entry point is callable in-process (the suite
    calls it), and a disposition set there would outlive the call in the
    caller's process with nothing to restore it. ``_install_sighup_ignore`` says
    the same about the runtime's entry.
    """
    parsed = build_parser().parse_args()
    # THE JOB LOG IS THIS PROCESS'S STDERR, so console logging is what makes the
    # worker's diagnostics reach the record a person reads — including the
    # one-shot SIGHUP survival line, which was silently discarded before this
    # because the root logger's default level is WARNING and nothing configured a
    # handler (review round 1, MINOR-1). Mirrors ``cli.main``'s own call; the
    # spawner already redirects this stderr into the job log (``exec_mode``).
    from local_operator.logger import configure_cli_logging

    configure_cli_logging()
    try:
        code = run(parsed)
    except Exception as exc:  # noqa: BLE001 — a log file is the only surface
        sys.stderr.write(f"exec_worker error: {exc}\n")
        code = 1
    if getattr(parsed, "job_id", None):
        try:
            from local_operator.exec_mode import update_job_exit

            update_job_exit(parsed.job_id, code)
        except Exception:  # noqa: BLE001 — best-effort ledger
            pass
    # Flush before returning: the spawner owns this file's lifetime and the
    # process may be reaped right after exit.
    sys.stdout.flush()
    sys.stderr.flush()
    return code


if __name__ == "__main__":
    # The Linux comm axis (see :func:`procname.brand_this_process`): macOS names
    # this worker from the image its parent exec'd it through, Linux has no such
    # image, so the process has to name itself. Called in the ``__main__``
    # branch rather than inside ``main()`` because ``main()`` is callable
    # in-process (the suite calls it), and a comm set on the CALLER's thread
    # would outlive the call — the same reason ``_install_sighup_ignore`` sits
    # where it does.
    from local_operator import procname

    procname.brand_this_process()
    sys.exit(main())

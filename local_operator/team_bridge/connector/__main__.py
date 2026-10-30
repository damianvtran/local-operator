"""``python -m local_operator.team_bridge.connector`` — the always-on process.

START-UP ORDER (each step exists because getting it wrong is a real failure):

1. Parse ``--config-dir``. No defaults that could silently target the
   operator's live config from a test: the caller names the root.
2. Read ``RADIENT_API_BASE_URL`` ONCE here — the value is restart-scoped per
   §13.3; nothing below re-reads the environment.
3. Resolve the committed owner (full namespace) and take the machine-wide
   connector lock BEFORE any network call. Two connectors on one account is
   the defect the lock exists to refuse; the loser exits quietly with its
   own message. With NO owner yet the process WAITS in-process (the loop's
   owner watch: a ``stat`` per poll, an owner read per bounded interval)
   instead of exiting: an exit 0 under ``KeepAlive: true`` was a whole
   interpreter relaunch about every 10 s (D2b).
4. Run the loop. The access provider is the H1 client's own ``current_access``
   (the owner-pinned OAuth row — the ordinary sign-in path; never a
   model-key cascade). When the loop hands back a DIFFERENT committed owner
   (a re-enrolment while running), re-arm on it in-process under the same
   lock with a client pinned to the new credential: no restart needed, and
   none of the refusal codes ever end the process (review R4).
5. Hand the loop the config-rooted credential store. Screening runs a model,
   and the model runner resolves its provider key from this store (rows,
   then process env). Starting with none held every message as
   ``provider_error`` before any screen (live E2E F4).

6. Stop CLEANLY on SIGTERM/SIGHUP (see ``_supervise``). launchd sends
   SIGTERM on a bootout, a logout and a reboot; with no handler the default
   action killed the process mid-spine, the spine's ``finally:
   release_claim`` never ran, and the message's claim stayed ``held`` for
   the whole lease (PR #3038 review R2-3).

This entry point does NOT install services; that is connector/install.py's
separate, guarded lifecycle (D2a), never a connector's own start path.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("local_operator.team_bridge.connector")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local_operator.team_bridge.connector")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--credential-id", type=int)
    parser.add_argument("--account-id")
    parser.add_argument("--tenant-id")
    parser.add_argument("--endpoint-id")
    parser.add_argument("--base-url")
    return parser


#: How long a stop signal lets the in-flight iteration finish on its own (a
#: spine already past screening usually completes and releases in well under
#: this) before the work is cancelled. Cancellation is what bounds the rest:
#: the spine's ``finally: release_claim`` runs on ``CancelledError``.
STOP_GRACE_S = 5.0
#: How long the cancelled work gets to unwind before the process exits
#: regardless. GRACE + CANCEL stays well inside launchd's default
#: ``ExitTimeOut`` (20 s), after which launchd SIGKILLs — the very state the
#: handler exists to avoid. Both are asyncio timers, so the real bound is
#: GRACE + CANCEL plus the longest SYNCHRONOUS step running on the loop thread
#: when they fire (a SQLite busy wait, ≤ 2 s); every such step is bounded. A
#: claim still held at exit is the loop's ``stale_claim`` park to recover.
STOP_CANCEL_S = 5.0
#: The stop signals: SIGTERM (launchd/systemd stop, reboot) and SIGHUP (the
#: controlling terminal went away). SIGINT keeps Python's KeyboardInterrupt.
_STOP_SIGNALS = tuple(
    sig for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if sig
)

#: How often a RUNNING connector compares its OWN build with the install pointer.
#:
#: WHY THIS EXISTS AT ALL. A connector is always-on (``KeepAlive: true``), so it
#: outlives every update that moved ``current`` under it: an update flips the
#: pointer, the running process keeps serving the generation it was exec'd from,
#: and nothing in the process asks whether that is still the build anyone
#: intended. Measured 2026-10-30: pid 48311, generation 20261023T142851Z-0.86.1,
#: held ``~/.local-operator/team_bridge/connector.lock`` for seven days while the
#: supervised job was refused 2,826 times — nothing inbound was admitted for
#: days, and the machine looked like a network problem.
#:
#: A minute is the loop's own backoff order, and it bounds how long a stale
#: process can hold the account's singleton after a flip.
SELF_CHECK_INTERVAL_S = 60.0

#: How long this process must have been UP before its own answer may stop it.
#:
#: THE FLAP GUARD, and the reason there is no check before the lock. A connector
#: started DURING a flip can read a pointer that moved a moment after its own
#: exec — the ordinary mixed-generation window of a few seconds — and stopping on
#: that read would turn a working connector into a restart loop (launchd's
#: ``ThrottleInterval`` would pace it, not prevent it). Five minutes is well past
#: any flip and cheap to wait for an always-on process.
SELF_CHECK_MIN_UPTIME_S = 300.0

#: ``time.monotonic()`` at import — a process start for every way this module is
#: run (``python -m``, a launched binary). A test shrinks the uptime gate by
#: patching this, not by sleeping.
_PROCESS_STARTED_S = time.monotonic()

#: The value ``XPC_SERVICE_NAME`` carries in a process launchd did NOT start:
#: macOS puts this in every interactive shell's environment, so "the variable is
#: set" alone would read a terminal-started connector as supervised.
_INTERACTIVE_SHELL_XPC = "0"


def _is_supervised() -> bool:
    """Whether a service manager owns this process's lifetime.

    THE EXIT IS ONLY SAFE WITH A SUPERVISOR. A self-check that sets ``stop``
    ends the process; under launchd (``KeepAlive: true``) or a systemd user unit
    (``Restart=always``) that is a restart onto the current generation, which is
    the whole point. In a terminal it is a connector that silently vanished.

    Each marker is the supervisor's OWN, not an inference:

    * macOS: launchd stamps ``XPC_SERVICE_NAME`` with the job's label on every
      process it spawns. The interactive-shell value (``0``) means no job, so a
      foreground run is excluded rather than trusted.
    * Linux: systemd sets ``INVOCATION_ID`` per unit invocation; nothing else on
      the host sets it.
    * Windows: never. There are no generations there (the layout is POSIX-only),
      so the question cannot arise; answering ``False`` keeps the check inert.
    """

    from local_operator.procstate import is_windows

    if is_windows():
        return False
    if os.environ.get("INVOCATION_ID"):
        return True
    name = os.environ.get("XPC_SERVICE_NAME", "")
    return bool(name) and name != _INTERACTIVE_SHELL_XPC


def _own_generation_is_stale() -> bool | None:
    """This process's own staleness, through the ONE comparison the product has.

    Deliberately ``update.generation_staleness_of_process`` rather than a second
    comparator: the installer repair, the surfaces and this process must not
    drift about what "older" means. TRI-STATE, and only ``True`` acts — ``None``
    (no generation in this argv, or an unreadable pointer) and ``False`` both
    mean "no move".
    """

    from local_operator.update import generation_staleness_of_process

    return generation_staleness_of_process(os.getpid())


async def _watch_own_generation(stop: asyncio.Event) -> None:
    """Stop a supervised connector that is provably serving a superseded build.

    Sets the SAME ``stop`` the signal handlers set, so the graceful path in
    ``_supervise`` runs unchanged: no new iteration, the in-flight one finishes
    inside ``STOP_GRACE_S``, the lock and the credential store close in
    ``_main_async``'s ``finally``, and the process exits 0 — which under
    ``KeepAlive`` is a restart, not a crash.

    THE PROBE FORKS ``ps``, so it runs in the default executor: a read that
    blocked the loop for tens of milliseconds every minute would delay every
    poll, pause and claim the loop is holding. Uptime is checked BEFORE the probe
    for the same reason it exists at all (see ``SELF_CHECK_MIN_UPTIME_S``).

    Unsupervised, it warns ONCE and keeps running: the line is what tells an
    operator their foreground connector is on an old build, and repeating it
    every minute would bury everything else in the log.
    """

    loop = asyncio.get_running_loop()
    supervised = _is_supervised()
    warned = False
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=SELF_CHECK_INTERVAL_S)
            return
        except TimeoutError:
            pass
        if time.monotonic() - _PROCESS_STARTED_S < SELF_CHECK_MIN_UPTIME_S:
            continue
        if await loop.run_in_executor(None, _own_generation_is_stale) is not True:
            continue
        if supervised:
            # One content-free line, in this file's vocabulary: the stage that
            # decided and the closed code that says why.
            logger.info("bridge connector stopping: stage=stop code=stale_generation")
            stop.set()
            return
        if not warned:
            warned = True
            logger.warning("bridge connector stale generation: stage=run code=stale_generation")


#: One rotating file is bounded at (BACKUP_COUNT + 1) x MAX_BYTES = 1.5 MiB.
_LOG_MAX_BYTES = 256 * 1024
_LOG_BACKUP_COUNT = 5


class _ClassNameOnlyFormatter(logging.Formatter):
    """Persist an exception as its CLASS name, never its message or traceback.

    Several connector sites log ``exc_info=True``; a traceback carries
    ``str(exception)`` (which can echo a body, a name or a token) and frame
    paths under the user's home. The records are therefore only content-free
    if this file's formatter enforces it. The record is cloned so other
    handlers (and the cached ``exc_text``) still see the full exception.
    """

    def format(self, record: logging.LogRecord) -> str:
        clone = copy.copy(record)
        clone.exc_text = None
        clone.stack_info = None
        return super().format(clone)

    def formatException(self, ei: Any) -> str:  # noqa: N802 - logging.Formatter API
        return f"exc={ei[0].__name__}" if ei and ei[0] is not None else ""


def _configure_logging(config_dir: Path) -> logging.Handler | None:
    """Send the ``local_operator.team_bridge`` loggers to a bounded private file.

    WHY: this process runs under launchd with stdout/stderr on ``/dev/null`` and
    configured no handler, so every ``logger.warning`` in the receive path went
    to Python's last-resort stderr handler and vanished. A receive outage ran
    for hours with nothing to read. The records are already content-free by
    construction at most call sites (closed codes, stage names, exception CLASS
    names), and the formatter ENFORCES it for the ones that log ``exc_info``;
    this gives them somewhere to land. Attached to the package logger rather than
    the root so no third-party record (httpx request lines carrying URLs) is
    captured. Returns the handler (for teardown/tests), or ``None`` when the
    file cannot be opened: no log is a degraded connector, a traceback at start
    is a dead one.
    """

    from local_operator.logger import DEFAULT_LOG_FORMAT, _PrivateRotatingFileHandler
    from local_operator.team_bridge.connector.install import log_path

    target = log_path(config_dir)
    try:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        handler = _PrivateRotatingFileHandler(
            target, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError:
        return None
    handler.setFormatter(_ClassNameOnlyFormatter(DEFAULT_LOG_FORMAT))
    package_logger = logging.getLogger("local_operator.team_bridge")
    package_logger.addHandler(handler)
    if package_logger.level == logging.NOTSET or package_logger.level > logging.INFO:
        package_logger.setLevel(logging.INFO)
    return handler


async def _main_async(args: argparse.Namespace, stop: asyncio.Event | None = None) -> int:
    from local_operator.providers.auth_store import AuthStore
    from local_operator.team_bridge.client import (
        DEFAULT_RADIENT_API_BASE_URL,
        TeamBridgeClient,
    )
    from local_operator.team_bridge.connector import loop as _loop
    from local_operator.team_bridge.connector.lock import (
        ConnectorLock,
        ConnectorLockBusy,
    )
    from local_operator.team_bridge.store import Store

    config_dir = Path(args.config_dir)
    _configure_logging(config_dir)
    # Restart-scoped transport root: read HERE, once.
    base_url = args.base_url or DEFAULT_RADIENT_API_BASE_URL

    store = Store(config_dir)
    if args.credential_id and args.account_id and args.tenant_id and args.endpoint_id:
        # A pinned namespace stays pinned across re-arms: only that exact
        # owner's generation can move under it (a credential or endpoint
        # change is a different namespace this invocation was not given).
        def owner_source() -> Any:
            return store.resolve_owner(
                credential_id=args.credential_id,
                account_id=args.account_id,
                tenant_id=args.tenant_id,
                endpoint_id=args.endpoint_id,
                canonical_base_url=base_url,
            )

    else:
        owner_source = store.active_owner
    owner = owner_source()
    if owner is None:
        # Content-free and once: the state persists until an enrolment
        # commits, and the watch below re-checks it without logging per tick.
        logger.info("bridge connector idle: stage=idle code=no_owner")
    while owner is None:
        if stop is not None and stop.is_set():
            return 0
        await _loop.idle_until_owner_changes(
            None, config_dir=config_dir, owner_source=owner_source, stop=stop
        )
        owner = owner_source()

    lock = ConnectorLock(config_dir)
    try:
        lock.acquire()
    except ConnectorLockBusy as busy:
        logger.warning("bridge connector waiting for singleton: %s", busy)
        # ONE reap attempt before giving up: a connector left on a build the
        # pointer has left keeps this lock for its whole life and nothing else
        # can ever take it. The proofs (and every refusal that keeps us off
        # someone else's process) live in `reap`; this call site decides only
        # that one attempt is made, once per process start.
        from local_operator.team_bridge.connector import reap as _reap

        outcome = await _reap.reap_stale_holder(config_dir)
        if outcome.kind == "reaped":
            try:
                lock.acquire()
            except ConnectorLockBusy:
                # It came back, or a third process took it in the window. There
                # is nothing further to do from here; the supervisor's retry is
                # the next attempt.
                return 75
        else:
            # Task Scheduler retries failures, not clean exits. A foreground
            # owner may relinquish this lock later; exit nonzero so the logon
            # receiver cannot remain permanently stopped after losing that race.
            return 75

    # F4 (live E2E): the store the screener reads its provider key from. The
    # explicit ``db_path`` is load-bearing — ``config_dir`` alone feeds only
    # the env-override tier and would leave the DATABASE at the ambient home,
    # so a desktop login under this config root would be invisible. A PLAIN
    # store, not the mesh-aware ``build_auth_store``: the same instance backs
    # the screening cycle's live-identity check, and the bridge client is
    # deliberately never served a bearer lent by a paired device (see
    # ``TeamBridgeClient._auth_store``). Values are never read or logged here.
    auth_store = AuthStore(db_path=config_dir / "auth.db", config_dir=config_dir)
    try:
        while owner is not None:
            # A fresh client per owner: it pins ``credential_id``, so a
            # re-enrolment onto a new credential row must not keep serving
            # the old row's bearer.
            client = TeamBridgeClient(
                config_dir,
                base_url=base_url,
                credential_id=owner.credential_id,
                # The connector's own store: a private one per client leaked an
                # auth.db connection per message (see TeamBridgeClient._auth_store).
                store=auth_store,
            )

            async def access_provider(client: TeamBridgeClient = client) -> Any:
                return await client.current_access()

            owner = await _loop.run_forever(
                owner,
                config_dir=config_dir,
                auth_store=auth_store,
                access_provider=access_provider,
                owner_source=owner_source,
                # Set by a stop signal: no NEW iteration starts, and every
                # pause/idle wakes at once. Returns ``None``, ending this loop.
                stop=stop,
            )
    finally:
        auth_store.close()
        lock.release()
    return 0


def _install_windows_stop() -> bool:
    """Windows: turn ``SIGBREAK`` into the same clean stop ``SIGINT`` already gets.

    ``loop.add_signal_handler`` does not exist on the Windows Proactor loop
    (``procstate.supports_loop_signals``), so a POSIX-style handler cannot be
    installed there. ``SIGINT`` (Ctrl+C) already surfaces as
    ``KeyboardInterrupt`` through ``asyncio.run``, which cancels the main task,
    so ``_main_async``'s ``finally`` releases the machine lock and closes the
    credential store. ``SIGBREAK`` (Ctrl+Break, and what a console-group stop
    sends) has no handler by default and terminates the process without any of
    that. Raising ``KeyboardInterrupt`` from a ``signal.signal`` handler puts it
    on the SAME path. A Task Scheduler ``/End`` terminates the process outright
    and runs no handler at all; the kernel drops the lock on exit, which is why
    the lock design never depends on a clean stop.

    Deliberately ADDITIVE and Windows-only: the POSIX SIGTERM/SIGHUP stop
    sequence belongs to ``_install_stop_handlers``/``_supervise`` and is not
    touched here. Returns
    whether a handler was installed.
    """

    import os
    import signal

    sigbreak = getattr(signal, "SIGBREAK", None)
    if os.name != "nt" or sigbreak is None:
        return False

    def _stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(sigbreak, _stop)
    except (OSError, RuntimeError, ValueError):
        # Not the main thread, or a host that refuses it: Ctrl+C still works.
        return False
    return True


def _install_stop_handlers(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> list[int]:
    """Route the stop signals to ``stop``; return the ones actually installed.

    ``add_signal_handler`` runs the callback ON the loop (never inside an
    arbitrary frame the way a ``signal.signal`` handler would), so setting an
    ``asyncio.Event`` from it is safe. It is UNIX-only: a platform that refuses
    it keeps the old default behaviour, which the ``stale_claim`` park still
    recovers — one claim lease later, and at-least-once (see loop.py).
    """

    installed: list[int] = []
    for sig in _STOP_SIGNALS:
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        installed.append(sig)
    return installed


async def _supervise(args: argparse.Namespace) -> int:
    """Run the connector until it returns or a stop signal arrives.

    THE STOP SEQUENCE (R2-3): the signal sets ``stop`` — the loop starts no
    new iteration and its pauses wake. The in-flight iteration gets
    ``STOP_GRACE_S`` to finish on its own (its spine reaches its ``finally``
    and releases its claim), then it is CANCELLED, which runs that same
    ``finally`` from the ``await`` it is parked on (screening, the long poll,
    the handoff). The machine lock and the credential store close in
    ``_main_async``'s own ``finally``. Exit is 0: a requested stop is not a
    failure, and ``KeepAlive`` should not read it as a crash.
    """

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    installed = _install_stop_handlers(loop, stop)
    work = asyncio.create_task(_main_async(args, stop))
    stopped = asyncio.create_task(stop.wait())
    watch = asyncio.create_task(_watch_own_generation(stop))
    try:
        await asyncio.wait({work, stopped}, return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            return work.result()
        if not watch.done():
            # The stop came from a signal; the self-check logs its own reason.
            logger.info("bridge connector stopping: stage=stop code=signal")
        await asyncio.wait({work}, timeout=STOP_GRACE_S)
        if not work.done():
            work.cancel()
            await asyncio.wait({work}, timeout=STOP_CANCEL_S)
        if not work.done():
            # The cancellation never landed: something is awaiting outside
            # the task's reach (an executor future, or a callee swallowing
            # ``CancelledError``). Waiting longer only ends in launchd's
            # SIGKILL; leave now — the kernel drops the flock and the loop's
            # stale-claim park recovers any claim still held.
            logger.warning("bridge connector stopping: stage=stop code=unwind_timeout")
            logging.shutdown()
            os._exit(0)
        if not work.cancelled() and work.exception() is not None:
            # Content-free, like every connector line: the class name only.
            logger.warning(
                "bridge connector stopping: stage=stop code=%s",
                type(work.exception()).__name__,
            )
        logger.info("bridge connector stopped: stage=stop code=clean")
        return 0
    finally:
        stopped.cancel()
        watch.cancel()
        if watch.done() and not watch.cancelled():
            # Retrieved rather than propagated: a self-check must never turn a
            # clean stop into a failure. Without this, asyncio would print the
            # task's exception as an unhandled task error instead.
            watch.exception()
        for sig in installed:
            loop.remove_signal_handler(sig)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _install_windows_stop()
    try:
        return asyncio.run(_supervise(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

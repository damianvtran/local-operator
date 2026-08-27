"""
Centralized logger configuration for the local_operator package.

Two destinations, chosen by the entry point rather than by import order:

- :func:`configure_console_logging` installs the single stderr handler the
  headless entry points (CLI, ``exec``, the plain REPL, the server) have always
  had. It is an EXPLICIT call. It used to happen as a side effect of importing
  this module and of importing ``local_operator.helpers``, which meant the
  effective configuration depended on which module the interpreter happened to
  reach first and no entry point could opt out — the reason the TUI could not
  turn console logging off.
- :func:`file_logging` swaps that handler for a bounded rotating file for the
  duration of a ``with`` block. The full-screen TUI owns the terminal; a log
  record written to stderr while Textual is painting lands on top of the frame
  and corrupts the display until the next full repaint.

Usage:
    from local_operator.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Message")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from local_operator.paths import ensure_log_dir, log_dir

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

#: The package's own console/file format.
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

#: The format the ``local-operator`` console script has emitted since the
#: import-time ``basicConfig`` in ``helpers.py`` set it. Kept verbatim, and kept
#: separate from :data:`DEFAULT_LOG_FORMAT`, because the two entry points have
#: always disagreed: the CLI shows ``time - LEVEL - message`` at INFO, the
#: server shows ``time [LEVEL] logger: message`` at ``LOG_LEVEL``. Unifying them
#: is a visible change to every scripted consumer of the CLI's stderr, so it is
#: not smuggled in with a logging-destination fix.
CLI_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

#: Name of the rotating log file inside :func:`local_operator.paths.log_dir`.
LOG_FILE_NAME = "local-operator.log"

#: Rotation bound. 2 MiB holds roughly 15k records — several days of ordinary
#: interactive use, and enough of a crashing session to be worth reading — and
#: four rotations keep the previous few sessions around after a long one. The
#: product of the two is the HARD ceiling: 2 MiB x (4 + 1) = 10 MiB, and
#: ``RotatingFileHandler`` reaps beyond it, so an agent stuck in a logging loop
#: overnight costs 10 MiB rather than a full disk.
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 4

#: The ceiling stated above, exported so callers and tests assert against one
#: number instead of recomputing the product.
LOG_TOTAL_MAX_BYTES = LOG_MAX_BYTES * (LOG_BACKUP_COUNT + 1)

#: Path of the log file currently receiving records, or ``None`` when nothing
#: is routed to a file. Read by the TUI's ``/help`` footer so the user can find
#: the file without being told the path on every launch.
_current_log_file: Optional[Path] = None

#: True while a :func:`file_logging` block owns the root logger. Nested blocks
#: must not stack a second handler on the same file (duplicate records) nor
#: hand the console back when the inner one exits.
_file_logging_active = False


def _get_log_level() -> int:
    """
    Get the log level from the LOG_LEVEL environment variable.
    Defaults to logging.WARNING if not set or invalid.
    """
    level_str = os.environ.get("LOG_LEVEL", "WARNING").upper()
    return _LOG_LEVELS.get(level_str, logging.WARNING)


def configure_console_logging(
    level: Optional[int] = None,
    fmt: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Route the root logger to stderr, replacing whatever was there.

    Always replaces rather than deferring like ``basicConfig`` does. The
    no-op-if-already-configured rule is what made the previous arrangement
    unpredictable: the winner was whichever module imported first, and
    ``generate_openapi.py``'s own ``basicConfig`` had silently been dead for as
    long as it had imported ``server.app`` on the line above it.

    STDERR, never stdout. stdout is a DATA channel: ``exec --json`` writes one
    JSON event per line there, and any log record interleaved into it corrupts
    the stream for a strict per-line ``json.loads`` consumer (httpx logs an INFO
    line per request, which is exactly the traffic an agent run generates).
    """
    resolved = _get_log_level() if level is None else level

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(handler)
    root_logger.setLevel(resolved)

    # Quieten noisy HTTP client libraries used by the provider wire clients.
    for lib_logger in ("requests", "urllib3", "httpx", "httpcore"):
        logging.getLogger(lib_logger).setLevel(resolved)


def configure_cli_logging() -> None:
    """Console logging exactly as the ``local-operator`` script has always had it.

    INFO regardless of ``LOG_LEVEL`` — the old import-time ``basicConfig`` in
    ``helpers.py`` passed a literal ``logging.INFO``, so the variable never
    reached the CLI and raising it here would start printing records to users
    who set ``LOG_LEVEL`` for the server.
    """
    configure_console_logging(level=logging.INFO, fmt=CLI_LOG_FORMAT)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a logger with the given name, configured with the centralized settings.

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)


def current_log_file() -> Optional[Path]:
    """The log file records are being written to, or ``None``."""
    return _current_log_file


def console_is_silenced() -> bool:
    """True while a :func:`file_logging` block owns the terminal.

    The one honest signal that ANOTHER surface — in practice the Textual TUI —
    is painting the screen and reading the keyboard. Code that would otherwise
    ``print`` to stderr or read stdin has to consult this first: a stray write
    lands inside a frame and stays there until the next full repaint, and a
    stray ``input()`` competes with Textual's driver for the same file
    descriptor, so the two readers split the user's keystrokes between them.

    Deliberately phrased about the CONSOLE rather than about the TUI: the
    contract is "something else owns this terminal", which is exactly what the
    file-logging window means, and it stays correct if another full-screen
    front end is added later.
    """
    return _file_logging_active


def _is_console_handler(handler: logging.Handler) -> bool:
    """True when ``handler`` writes to the terminal the TUI is drawing on.

    Identity against ``sys.stdout``/``sys.stderr`` is not enough. Textual
    replaces both for the duration of ``run_async`` (``redirect_stdout`` /
    ``redirect_stderr``), but a ``StreamHandler`` built BEFORE that swap holds a
    direct reference to the original file object and keeps writing to the real
    terminal straight through the redirect. That is precisely the handler the
    import-time ``basicConfig`` installed, and precisely why the redirect never
    saved the frame. So compare against the pristine ``sys.__stdout__`` /
    ``sys.__stderr__`` as well, then fall back to the file descriptor.
    """
    if isinstance(handler, logging.FileHandler):
        # A FileHandler is a StreamHandler subclass, but its stream is a file.
        return False
    stream = getattr(handler, "stream", None)
    if stream is None:
        return False
    if stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        return True
    try:
        return stream.fileno() in (1, 2)
    except Exception:  # noqa: BLE001 — an in-memory stream has no descriptor
        return False


def _all_loggers() -> Iterator[logging.Logger]:
    """The root logger plus every logger instantiated so far."""
    yield logging.getLogger()
    for candidate in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(candidate, logging.Logger):
            yield candidate


@dataclass
class _ConsoleSilence:
    """What :func:`file_logging` took away, so it can be put back exactly."""

    removed: list[tuple[logging.Logger, logging.Handler]] = field(default_factory=list)
    last_resort: Optional[logging.Handler] = None
    raise_exceptions: bool = True
    original_add_handler: Optional[Callable[..., None]] = None


def _detach_console_handlers() -> _ConsoleSilence:
    """Remove every terminal-bound handler and block new ones being installed."""
    state = _ConsoleSilence(
        last_resort=logging.lastResort,
        raise_exceptions=logging.raiseExceptions,
        original_add_handler=logging.Logger.addHandler,
    )

    for logger_obj in _all_loggers():
        for handler in list(logger_obj.handlers):
            if _is_console_handler(handler):
                logger_obj.removeHandler(handler)
                state.removed.append((logger_obj, handler))

    # `lastResort` is a WARNING-level stderr handler the logging module uses
    # when a record reaches a logger with no handlers at all. If the log
    # directory could not be created there IS no handler, and every warning
    # would go straight onto the frame through this back door.
    logging.lastResort = None

    # A handler that fails to write (full disk, revoked permissions mid-run)
    # normally dumps "--- Logging error ---" plus a traceback onto stderr from
    # inside `Handler.handleError`. That is a log-about-logging painted over
    # the UI; the file handler's own failures must stay silent here.
    logging.raiseExceptions = False

    original_add_handler = logging.Logger.addHandler

    def guarded_add_handler(self: logging.Logger, handler: logging.Handler) -> None:
        """Drop terminal-bound handlers installed while the TUI owns the screen.

        Removing the handlers that exist at startup is not sufficient on its
        own: a library imported lazily mid-session (a provider SDK, an MCP
        client) can call ``basicConfig`` or attach its own ``StreamHandler``,
        and the very first record it logs then lands on the frame. Refusing the
        installation is the only point where that is catchable.
        """
        if _is_console_handler(handler):
            return
        original_add_handler(self, handler)

    logging.Logger.addHandler = guarded_add_handler  # type: ignore
    return state


def _restore_console_handlers(state: _ConsoleSilence) -> None:
    """Undo :func:`_detach_console_handlers` so later phases log normally."""
    if state.original_add_handler is not None:
        logging.Logger.addHandler = state.original_add_handler  # type: ignore
    logging.lastResort = state.last_resort
    logging.raiseExceptions = state.raise_exceptions
    for logger_obj, handler in state.removed:
        logger_obj.addHandler(handler)


def _open_rotating_handler(
    max_bytes: int,
    backup_count: int,
) -> tuple[Optional[logging.Handler], Optional[Path]]:
    """Open the bounded rotating handler, or ``(None, None)`` if impossible."""
    directory = ensure_log_dir()
    if directory is None:
        return None, None
    path = directory / LOG_FILE_NAME
    try:
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=False,
        )
    except OSError:
        # Same rule as `ensure_log_dir`: no log file is a degraded session, a
        # traceback on startup is a broken one.
        return None, None
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    # 0o600 to match the directory's 0o700 and `credentials.env`'s own mode:
    # the file carries prompt and error text from an interactive session.
    try:
        os.chmod(path, 0o600)
    except OSError:  # Windows and exotic filesystems; the log still works
        pass
    return handler, path


def _redirect_fd_stderr(logfd: int) -> Optional[int]:
    """Point the process's REAL stderr file descriptor (fd 2) at the log file.

    :func:`file_logging` and Textual's own ``redirect_stderr`` only swap the
    Python-level ``sys.stderr`` object and the logging handlers bound to it.
    That is enough for anything written *through Python*, but it does nothing
    about code below the interpreter that calls the raw ``write(2)`` syscall on
    file descriptor 2 directly. Those bytes never pass through ``sys.stderr``,
    so every Python-level redirect misses them and they land straight on top of
    whatever Textual has painted on the alternate screen, smearing the frame
    and the composer until the next full repaint.

    The concrete offender on macOS is libmalloc: under memory pressure (and on
    lock/unlock and sleep/wake, which churn the allocator the same way) it
    emits ``MallocStackLogging: can't turn off malloc stack logging because it
    was not enabled.`` by writing it to fd 2 itself, underneath CPython. We
    have zero ``malloc`` references in the tree — this is the OS, not us — which
    is exactly why no Python-level guard can catch it. Redirecting the OS-level
    fd 2 into the rotating log file does: the native bytes are captured where
    they can be grepped in ``local-operator.log`` (the operator's choice over
    ``/dev/null`` — these diagnostics correlate with real RAM-pressure events
    and are worth keeping), and the frame stays clean. This mirrors the fix in
    openai/codex PR #24459 ("prevent macos stderr from corrupting composer").

    macOS-only by design: Linux and Windows do not emit this, and moving fd 2
    out from under them would be needless risk for no benefit. Returns the
    saved original fd 2 (to restore on exit) or ``None`` when the guard did not
    install — off-darwin, or a syscall failed and the caller should leave fd 2
    untouched rather than send it somewhere invalid.

    Rotation trade-off: ``dup2(logfd, 2)`` makes fd 2 share the log file's
    *open file description* as it stands right now, independent of ``logfd``
    itself. When ``RotatingFileHandler`` rolls over it closes and reopens its
    own stream, but fd 2 keeps its reference to the install-time description, so
    native bytes emitted after a mid-session rotation keep flowing to the
    pre-rotation inode (the rolled ``.1`` file) until the block exits. That is
    accepted deliberately: the bytes are still captured, just in the rolled
    file, and mid-session rotation is rare. The alternative — re-pointing fd 2
    on every rollover — would couple this allocator guard to logging internals
    for a case that almost never happens.
    """
    # Guard: this defect is macOS-specific, so the fd surgery is too.
    if sys.platform != "darwin":
        return None
    try:
        # Preserve the real terminal (or whatever owns fd 2) so exit restores it
        # exactly, including on exception.
        saved_fd = os.dup(2)
    except OSError:
        return None
    try:
        os.dup2(logfd, 2)
    except OSError:
        # Could not redirect; do not leave a dangling saved fd behind.
        try:
            os.close(saved_fd)
        except OSError:
            pass
        return None
    return saved_fd


def _restore_fd_stderr(saved_fd: Optional[int]) -> None:
    """Undo :func:`_redirect_fd_stderr`, putting the original fd 2 back.

    Best-effort and non-raising in both syscalls: this runs in ``file_logging``'s
    ``finally``, where a teardown exception would mask the real one.
    """
    if saved_fd is None:
        return
    try:
        os.dup2(saved_fd, 2)
    except OSError:  # noqa: BLE001 — teardown must not raise
        pass
    try:
        os.close(saved_fd)
    except OSError:  # noqa: BLE001 — teardown must not raise
        pass


def _write_header(handler: logging.Handler, path: Path) -> None:
    """State the file's own path and bound as its first record.

    Emitted through ``handler.handle`` rather than a logger call so it appears
    regardless of the root level: at the default WARNING an ``info()`` here
    would be dropped and the file would start with no explanation of what it is
    or why it stops growing at 10 MiB.
    """
    record = logging.LogRecord(
        name=__name__,
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=(
            "logging to %s (rotating: %d bytes x %d backups, %d bytes total maximum); "
            "console output is suppressed while the TUI owns the terminal"
        ),
        args=(str(path), LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_TOTAL_MAX_BYTES),
        exc_info=None,
    )
    handler.handle(record)


@contextmanager
def file_logging(
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
) -> Iterator[Optional[Path]]:
    """Send log records to a bounded rotating file, and nowhere else, in-block.

    Yields the log file path, or ``None`` when no file could be opened. Console
    handlers are detached either way: with the TUI on screen, silence is the
    contract and a lost diagnostic is cheaper than a corrupted frame.

    Deliberately does NOT touch the root logger's level. The file is meant to
    receive exactly the records that would otherwise have been printed, so that
    "it did not appear on screen" and "it is not in the log" cannot diverge.

    Restores the previous handlers on exit, including on exception, so the
    plain REPL, ``exec`` and the server — which share this process when the CLI
    falls back — are unaffected once the TUI is gone.

    RE-ENTRANT: a nested block is a no-op that re-yields the active path. The
    CLI opens the window early, around scheduler startup, because the scheduler
    logs "Scheduler started" before the app paints; ``run_tui`` opens it too so
    the guarantee belongs to the TUI rather than to one of its callers. Without
    this, the inner block would attach a SECOND handler to the same file —
    every record duplicated — and its exit would restore the console early.
    """
    global _current_log_file, _file_logging_active

    if _file_logging_active:
        yield _current_log_file
        return

    state = _detach_console_handlers()
    handler, path = _open_rotating_handler(max_bytes, backup_count)
    root_logger = logging.getLogger()
    saved_stderr_fd: Optional[int] = None
    if handler is not None and path is not None:
        # Passes the guard installed above by design: `_is_console_handler`
        # rejects FileHandler, so the file handler is the one thing that can
        # still be attached while the TUI owns the screen.
        root_logger.addHandler(handler)
        _write_header(handler, path)
        _current_log_file = path
        # Reuse the file the handler just opened for the fd-level stderr guard
        # (see `_redirect_fd_stderr`): its underlying fd is our redirect target.
        # Only the OUTER `file_logging` block reaches here — the re-entrancy
        # early-return above means a nested block never re-installs the guard —
        # so its lifetime is tied to the same ownership window as the handler.
        # If no file could be opened (`handler is None`) fd 2 is left alone.
        # `stream` is a FileHandler concretion, not on the base Handler type,
        # so reach it defensively and treat a rolled/closed stream as "no fd".
        stream = getattr(handler, "stream", None)
        logfd: Optional[int] = None
        if stream is not None:
            try:
                logfd = stream.fileno()
            except (OSError, ValueError):
                logfd = None
        if logfd is not None:
            saved_stderr_fd = _redirect_fd_stderr(logfd)
    _file_logging_active = True
    try:
        yield path
    finally:
        _file_logging_active = False
        _current_log_file = None
        # Restore the real fd 2 first, before the handler that owns the redirect
        # target is closed, so the terminal is back the instant the TUI exits.
        _restore_fd_stderr(saved_stderr_fd)
        if handler is not None:
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
        _restore_console_handlers(state)


__all__ = [
    "CLI_LOG_FORMAT",
    "DEFAULT_LOG_FORMAT",
    "LOG_BACKUP_COUNT",
    "LOG_FILE_NAME",
    "LOG_MAX_BYTES",
    "LOG_TOTAL_MAX_BYTES",
    "configure_cli_logging",
    "configure_console_logging",
    "console_is_silenced",
    "current_log_file",
    "file_logging",
    "get_logger",
    "log_dir",
]

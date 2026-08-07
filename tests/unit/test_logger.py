"""Logging destinations: explicit console config, and the TUI's bounded file.

The defect these cover: a ``logging.basicConfig`` at module import installed a
stderr ``StreamHandler``, so every log record emitted while Textual owned the
terminal was painted across the frame. Two properties matter and neither is
"the message was formatted nicely" — that nothing reaches the terminal, and
that the file it goes to instead cannot grow without bound.
"""

from __future__ import annotations

import io
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.logger import (
    CLI_LOG_FORMAT,
    DEFAULT_LOG_FORMAT,
    LOG_BACKUP_COUNT,
    LOG_FILE_NAME,
    LOG_MAX_BYTES,
    LOG_TOTAL_MAX_BYTES,
    _is_console_handler,
    configure_cli_logging,
    configure_console_logging,
    current_log_file,
    file_logging,
)
from local_operator.paths import CONFIG_DIR_ENV


@pytest.fixture(autouse=True)
def restore_root_logger():
    """Give every test the root logger back exactly as it found it.

    The module under test mutates process-global state (root handlers,
    ``logging.lastResort``, ``logging.Logger.addHandler``). Leaking any of that
    into a sibling test would show up as an unrelated failure somewhere else in
    the session, which is the hardest kind of test bug to trace.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_last_resort = logging.lastResort
    saved_raise = logging.raiseExceptions
    saved_add_handler = logging.Logger.addHandler
    try:
        yield
    finally:
        logging.Logger.addHandler = saved_add_handler  # type: ignore
        logging.lastResort = saved_last_resort
        logging.raiseExceptions = saved_raise
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


@pytest.fixture
def log_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the log directory at a tmp dir and return the expected file path."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path / "logs" / LOG_FILE_NAME


def _attach_fake_console(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Install the exact handler shape the old import-time ``basicConfig`` did.

    ``sys.stderr`` is swapped for a buffer FIRST and the handler is built from
    it, so the handler holds the object the code under test must recognise as
    "the terminal". A handler over a bare ``StringIO`` would be silently exempt
    and the test would pass without exercising anything.
    """
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(CLI_LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return stream


# --- Importing must not configure anything ---------------------------------


def test_importing_helpers_does_not_configure_the_root_logger() -> None:
    """The regression that made the TUI unfixable: config as an import side effect.

    ``helpers`` is on the CLI's core import path, so its ``basicConfig`` ran for
    ``--version`` and for the TUI alike and no entry point could opt out. A
    fresh interpreter is the only honest way to assert this — the current one
    has already imported half the package.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging, local_operator.helpers, local_operator.logger;"
            "print(len(logging.getLogger().handlers))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "0"


# --- Console configuration is explicit and reproduces the historical shape ---


def test_configure_cli_logging_matches_the_historical_cli_console() -> None:
    """INFO on stderr with the legacy format — byte-for-byte what ``exec`` had.

    ``LOG_LEVEL`` is deliberately NOT consulted: the old ``basicConfig`` passed a
    literal ``logging.INFO``, so honouring the variable here would start
    printing records to anyone who set it for the server.
    """
    import os
    import sys

    os.environ["LOG_LEVEL"] = "CRITICAL"
    try:
        configure_cli_logging()
    finally:
        del os.environ["LOG_LEVEL"]

    root = logging.getLogger()
    assert root.level == logging.INFO
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr
    assert handler.formatter is not None
    assert handler.formatter._fmt == CLI_LOG_FORMAT


def test_configure_console_logging_replaces_rather_than_defers() -> None:
    """Unlike ``basicConfig``, a second call wins.

    ``basicConfig``'s no-op-when-configured rule is what made the old behaviour
    depend on import order; ``serve`` relies on the server's call overriding the
    CLI's, which only works if the call replaces.
    """
    configure_cli_logging()
    configure_console_logging(level=logging.ERROR)

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.ERROR
    assert root.handlers[0].formatter is not None
    assert root.handlers[0].formatter._fmt == DEFAULT_LOG_FORMAT


# --- file_logging: nothing on the terminal, everything in the file ----------


def test_file_logging_silences_the_console_and_captures_the_record(
    log_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    console = _attach_fake_console(monkeypatch)
    logger = logging.getLogger("local_operator.test")

    with file_logging() as path:
        assert path == log_home
        logger.warning("deliberate warning")
        try:
            raise RuntimeError("deliberate failure")
        except RuntimeError:
            logger.exception("boom")

    assert console.getvalue() == ""
    contents = log_home.read_text(encoding="utf-8")
    assert "deliberate warning" in contents
    assert "boom" in contents
    # The traceback, not just the message: a logged exception with the frames
    # stripped is the half of the record that would have been worth reading.
    assert "RuntimeError: deliberate failure" in contents
    assert "Traceback (most recent call last)" in contents


def test_file_logging_restores_the_console_on_exit(
    log_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``exec`` and the plain REPL can run in the same process after the TUI."""
    console = _attach_fake_console(monkeypatch)

    with file_logging():
        pass

    logging.getLogger("local_operator.test").warning("after the tui")
    assert "after the tui" in console.getvalue()


def test_file_logging_restores_the_console_after_an_exception(
    log_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    console = _attach_fake_console(monkeypatch)

    with pytest.raises(ValueError):
        with file_logging():
            raise ValueError("app crashed")

    logging.getLogger("local_operator.test").warning("still logging")
    assert "still logging" in console.getvalue()


def test_file_logging_blocks_a_console_handler_installed_mid_session(
    log_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A library that configures logging after the TUI starts must not get through.

    Detaching the handlers that exist at startup is not enough on its own: an
    SDK imported lazily mid-session calls ``basicConfig`` or attaches its own
    ``StreamHandler``, and its first record lands on the frame.
    """
    console = _attach_fake_console(monkeypatch)

    with file_logging():
        # Three shapes of the same failure: a bare StreamHandler on the
        # terminal, `basicConfig` reaching for the root logger, and a handler
        # built over something else then repointed at the terminal.
        logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))
        logging.basicConfig(level=logging.DEBUG)
        repointed = cast("logging.StreamHandler[Any]", logging.StreamHandler(io.StringIO()))
        repointed.stream = sys.stderr
        logging.getLogger("noisy.sdk").addHandler(repointed)

        logging.getLogger("noisy.sdk").warning("from a third party")

        leaked = [
            handler
            for logger_name in ("", "noisy.sdk")
            for handler in logging.getLogger(logger_name).handlers
            if _is_console_handler(handler)
        ]
        assert leaked == []

    assert console.getvalue() == ""
    assert "from a third party" in log_home.read_text(encoding="utf-8")


def test_file_logging_survives_an_uncreatable_log_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No log file is a degraded session; a crash on startup is a broken one.

    The console still goes quiet. With the TUI on screen, silence is the
    contract — a lost diagnostic is cheaper than a corrupted frame.
    """
    blocked = tmp_path / "cfg"
    blocked.mkdir()
    (blocked / "logs").write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(CONFIG_DIR_ENV, str(blocked))
    console = _attach_fake_console(monkeypatch)

    with file_logging() as path:
        assert path is None
        assert current_log_file() is None
        logging.getLogger("local_operator.test").warning("nowhere to go")

    assert console.getvalue() == ""


def test_file_logging_disarms_the_last_resort_handler(log_home: Path) -> None:
    """``logging.lastResort`` is a stderr handler that fires when none are left.

    It is the back door: if the file could not be opened there are no handlers
    at all, and every WARNING would be printed straight onto the frame by the
    logging module itself.
    """
    saved = logging.lastResort
    with file_logging():
        assert logging.lastResort is None
    assert logging.lastResort is saved


def test_file_logging_publishes_the_path_for_the_ui(log_home: Path) -> None:
    """The TUI's ``/help`` footer reads this; outside the block there is none."""
    assert current_log_file() is None
    with file_logging():
        assert current_log_file() == log_home
    assert current_log_file() is None


def test_nested_file_logging_neither_duplicates_nor_restores_early(
    log_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI opens the window around scheduler startup; ``run_tui`` opens it again.

    Both are deliberate — the guarantee belongs to the TUI, and the CLI needs
    it a moment earlier — so the inner block must be inert. A second handler on
    the same file would double every record, and an inner exit that restored
    the console would hand the terminal back while the app was still painting.
    """
    console = _attach_fake_console(monkeypatch)
    logger = logging.getLogger("local_operator.test.nested")

    with file_logging() as outer:
        with file_logging() as inner:
            assert inner == outer
        # The inner exit must not have given the console back.
        logger.warning("only once")
        assert current_log_file() == outer

    assert console.getvalue() == ""
    contents = log_home.read_text(encoding="utf-8")
    assert contents.count("only once") == 1


def test_log_file_names_itself_in_its_first_record(log_home: Path) -> None:
    """A log whose own path and bound are unstated is a log nobody can reason about."""
    with file_logging():
        pass

    first_line = log_home.read_text(encoding="utf-8").splitlines()[0]
    assert str(log_home) in first_line
    assert str(LOG_TOTAL_MAX_BYTES) in first_line


# --- The bound is real ------------------------------------------------------


def test_rotation_ceiling_is_the_documented_product() -> None:
    assert LOG_TOTAL_MAX_BYTES == LOG_MAX_BYTES * (LOG_BACKUP_COUNT + 1)


def test_rotation_reaps_old_files_and_holds_the_ceiling(log_home: Path) -> None:
    """Write far past the ceiling; the directory must not follow.

    Scaled-down bounds (16 KiB x 3) rather than the shipped 10 MiB, so the test
    costs milliseconds instead of writing ten megabytes — the property under
    test is the arithmetic of the rotation, which does not care about the
    magnitude. ``LOG_TOTAL_MAX_BYTES`` is pinned to the shipped constants by the
    test above.
    """
    max_bytes = 16 * 1024
    backup_count = 3
    ceiling = max_bytes * (backup_count + 1)
    logger = logging.getLogger("local_operator.test.rotation")
    payload = "x" * 512

    with file_logging(max_bytes=max_bytes, backup_count=backup_count):
        # 4x the ceiling of data, so anything that failed to reap is obvious.
        for index in range(4 * ceiling // len(payload)):
            logger.warning("%d %s", index, payload)

    log_dir_path = log_home.parent
    files = sorted(p.name for p in log_dir_path.iterdir())
    total = sum(p.stat().st_size for p in log_dir_path.iterdir())

    # backupCount rotations plus the live file, and nothing else left behind.
    assert files == [
        LOG_FILE_NAME,
        f"{LOG_FILE_NAME}.1",
        f"{LOG_FILE_NAME}.2",
        f"{LOG_FILE_NAME}.3",
    ]
    # RotatingFileHandler rolls when a record WOULD exceed maxBytes, so each
    # file can overshoot by at most one record; the slack is one record, not a
    # percentage of the ceiling.
    assert total <= ceiling + 2 * len(payload)

    # The newest data survived: reaping must drop the OLDEST file, not the live one.
    assert "x" in log_home.read_text(encoding="utf-8")

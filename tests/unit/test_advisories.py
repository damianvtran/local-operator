"""Advisory routing: the record always exists, the console line only for a person.

THE MEASURED FAILURE (2026-10-24): a `lop update` press inside PyPI's
propagation window failed with ``fresh PyPI latest is 0.88.4, not 0.89.0; retry
lop update``. The desktop app captures the child's stderr and diagnoses from its
HEAD, and two startup advisories (a config-drift notice and a duplicate-project
row notice) filled that head; the cause sat on line ~7 and never reached the
modal. These tests pin the fix at the seam that caused it: a handler that writes
to a non-terminal drops advisory records, while the record still reaches every
other handler — and a non-advisory warning is never dropped, because suppressing
real failures would be a worse defect than the one being fixed.
"""

from __future__ import annotations

import io
import logging

import pytest

from local_operator import advisories


class _Terminal(io.StringIO):
    """A stream a person is looking at: ``isatty`` says so."""

    def isatty(self) -> bool:  # noqa: D102 — the protocol is the whole class
        return True


def _logger_with_handler(stream: io.StringIO) -> logging.Logger:
    logger = logging.getLogger(f"tests.advisories.{id(stream)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    handler = logging.StreamHandler(stream)
    handler.addFilter(advisories.console_filter(stream))
    logger.addHandler(handler)
    return logger


def test_a_captured_stream_drops_the_advisory_but_the_record_survives(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pipe gets nothing; ``caplog`` (every other handler) gets the WARNING."""
    stream = io.StringIO()  # isatty() is False: this is a captured stream
    logger = _logger_with_handler(stream)

    with caplog.at_level(logging.WARNING, logger=logger.name):
        advisories.emit("config.yml lost 1 setting(s)", logger=logger)

    assert stream.getvalue() == ""
    records = [r for r in caplog.records if r.name == logger.name]
    assert [r.getMessage() for r in records] == ["config.yml lost 1 setting(s)"]
    assert records[0].levelno == logging.WARNING
    assert getattr(records[0], advisories.ADVISORY_ATTRIBUTE) is True


def test_a_terminal_still_shows_the_advisory(caplog: pytest.LogCaptureFixture) -> None:
    """A person watching a run keeps the line, once, from the same record."""
    stream = _Terminal()
    logger = _logger_with_handler(stream)

    with caplog.at_level(logging.WARNING, logger=logger.name):
        advisories.emit("config.yml lost 1 setting(s)", logger=logger)

    assert stream.getvalue() == "config.yml lost 1 setting(s)\n"
    assert [r.getMessage() for r in caplog.records if r.name == logger.name] == [
        "config.yml lost 1 setting(s)"
    ]


def test_a_warning_that_is_not_an_advisory_is_never_filtered() -> None:
    """The filter is keyed on the marker, never on the message or the logger.

    A genuine failure that quotes an advisory sentence, or any other WARNING on
    the same logger, must reach a captured stream — this is the line between
    "quiet diagnostic stream" and "suppressed errors".
    """
    stream = io.StringIO()
    logger = _logger_with_handler(stream)

    logger.warning("the update failed: config.yml lost 1 setting(s)")

    assert "the update failed" in stream.getvalue()


def test_the_console_filter_is_what_configure_cli_logging_installs() -> None:
    """The CLI's stderr handler carries the filter; a plain console config does not.

    Asserted on the configured objects rather than by calling ``lop``: the
    distinction IS the configuration, and a future refactor that drops the
    argument should fail here rather than in a modal on the operator's screen.
    """
    from local_operator.logger import configure_cli_logging, configure_console_logging

    def marked(level: int) -> logging.LogRecord:
        record = logging.LogRecord("x", level, __file__, 1, "advisory text", None, None)
        record.lop_advisory = True
        return record

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        configure_cli_logging()
        cli_handler = root.handlers[-1]
        # The bound stream is pytest's capture (not a tty), so the marked
        # record is vetoed while an ordinary warning passes.
        assert not cli_handler.filter(marked(logging.WARNING))
        plain = logging.LogRecord("x", logging.WARNING, __file__, 1, "failed", None, None)
        assert cli_handler.filter(plain)

        configure_console_logging()
        server_handler = root.handlers[-1]
        assert server_handler.filters == [], "a server's stderr IS its log destination"
        assert server_handler.filter(marked(logging.WARNING))
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def test_the_helper_emits_no_line_of_its_own(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The record is the only emission channel: no ``print`` beside the handler.

    A stray ``print`` would bypass the filter entirely (the pre-2026-10-23 call
    site printed AND logged, and every operator saw the paragraph twice), so the
    helper's own output is pinned to nothing.
    """
    stream = io.StringIO()
    logger = _logger_with_handler(stream)
    advisories.emit("one notice", logger=logger)

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""

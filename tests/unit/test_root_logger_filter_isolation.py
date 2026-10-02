"""The root-logger guard must undo a FILTER, not just a handler list.

``tests/conftest.py::restore_root_logger`` puts the process-global logging state
back between tests. Restoring the *list* of root handlers is enough for a handler
a test installed — the list assign drops it — but not for a filter a test
attached to a handler it did not install.

That gap has teeth because ``_pytest.logging`` builds its capture handlers ONCE
for the session (``LoggingPlugin.caplog_handler``/``report_handler``) and every
later test reuses the same object, and because ``local_operator.mcp.redaction``
attaches a record-REWRITING filter to every root handler the moment a credential
is resolved (``register()`` → ``attach()``). The rewrite clears ``exc_info`` in
place — it moves the rendered traceback to ``exc_text`` so a secret in it cannot
be re-interpolated by a downstream formatter — so from that resolve onward, in
that worker, every ``caplog.records`` entry carries ``exc_info=None``.

That is not hypothetical: it is why
``tests/unit/tui/test_projects_send.py::test_an_internal_fault_reaches_the_reader_as_a_sentence``
failed CI shard 2 tree-wide while passing when run alone, for as long as the
shuffled worker happened to schedule an MCP credential test first.

Both cases below drive the boundary INSIDE one test, through the guard the
suite's autouse fixture uses (``root_logger_guard``). Two tests could not do
this: pytest-xdist hands tests to workers individually and ``-n auto --dist
worksteal`` is this repo's own addopts, so a two-test pair splits across workers
and passes on the unfixed tree. Each case also asserts its own precondition, so
losing the attach (or the registration path) fails here loudly instead of
letting the case pass vacuously.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest

Guard = Callable[[], AbstractContextManager[None]]

#: Long enough to clear ``redaction.MIN_SCRUBBED_LENGTH``; a shorter value
#: returns from ``register()`` before it reaches ``attach()``.
_REGISTERED = "SENTINEL-FILTER-ISOLATION-40711"


def _emit_a_record_carrying_a_traceback() -> None:
    """Log exactly what the quick-send fault path logs, traceback included."""
    log = logging.getLogger("root-logger-filter-isolation")
    try:
        raise AttributeError("cannot unpack non-iterable coroutine object")
    except AttributeError:
        log.warning("quick-send resolve failed", exc_info=True)


def _carrying(handlers: list[logging.Handler]) -> list[str]:
    """Names of the handlers still carrying the redaction filter."""
    from local_operator.mcp import redaction

    return [type(h).__name__ for h in handlers if redaction._FILTER in h.filters]


def test_a_filter_attached_during_one_test_does_not_reach_the_next(
    root_logger_guard: Guard, caplog: pytest.LogCaptureFixture
) -> None:
    """Attach inside the boundary, then assert on what the next one sees."""
    from local_operator.mcp import redaction

    with root_logger_guard():
        # Stands in for everything a test body does before its boundary.
        redaction.register(_REGISTERED)
        try:
            assert _carrying(logging.getLogger().handlers), (
                "registration must attach _FILTER to the root handlers, or this test "
                "proves nothing about the guard"
            )
        finally:
            redaction.unregister(_REGISTERED)

    # ...the boundary has been crossed...
    assert not _carrying(
        logging.getLogger().handlers
    ), "the guard left the redaction filter on a handler the next test reuses"
    _emit_a_record_carrying_a_traceback()
    assert any(record.exc_info for record in caplog.records), [
        (record.getMessage(), record.exc_info, record.exc_text) for record in caplog.records
    ]


def test_the_guard_also_drops_a_filter_that_predates_its_snapshot(
    root_logger_guard: Guard, caplog: pytest.LogCaptureFixture
) -> None:
    """A registration landing BEFORE the snapshot must not survive either.

    Stands in for a module- or session-scoped fixture that resolves a credential:
    the filter is already on the root handlers when the guard takes its snapshot,
    so restoring the snapshot alone would record it as "as I found it" and hand
    it to every later test of that worker.
    """
    from local_operator.mcp import redaction

    redaction.register(_REGISTERED)
    try:
        with root_logger_guard():
            assert _carrying(
                logging.getLogger().handlers
            ), "precondition: the filter must already be attached when the snapshot is taken"
        assert not _carrying(
            logging.getLogger().handlers
        ), "the guard restored a pre-existing filter instead of dropping it"
        _emit_a_record_carrying_a_traceback()
        assert any(record.exc_info for record in caplog.records), [
            (record.getMessage(), record.exc_info, record.exc_text) for record in caplog.records
        ]
    finally:
        redaction.unregister(_REGISTERED)

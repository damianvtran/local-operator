"""The root-logger isolation guard must undo a FILTER, not just a handler list.

``tests/conftest.py::restore_root_logger`` puts the process-global logging state
back between tests. It restored the *list* of root handlers, which is enough for
a handler a test installed — the list assign drops it — but not for a filter a
test attached to a handler it did not install.

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

The pair below is deliberately order-dependent — that IS the property under
test, and it is deterministic because pytest runs a module's tests in file order
and CI shards by file. ``test_a`` asserts its own precondition, so removing the
attach (or the registration path) fails loudly here instead of letting the pair
pass vacuously.
"""

from __future__ import annotations

import logging

import pytest

#: Long enough to clear ``redaction.MIN_SCRUBBED_LENGTH``; a shorter value
#: returns from ``register()`` before it reaches ``attach()``.
_REGISTERED = "SENTINEL-FILTER-ISOLATION-40711"


def test_a_resolving_a_credential_attaches_the_scrubber_to_every_root_handler() -> None:
    """Register a credential, as resolving an MCP one does in production."""
    from local_operator.mcp import redaction

    redaction.register(_REGISTERED)
    try:
        attached = [h for h in logging.getLogger().handlers if redaction._FILTER in h.filters]
        assert attached, (
            "registration must attach _FILTER to the root handlers, or the pair below "
            "proves nothing about the guard"
        )
    finally:
        redaction.unregister(_REGISTERED)


def test_b_the_next_test_still_receives_exc_info(caplog: pytest.LogCaptureFixture) -> None:
    """The filter attached by ``test_a`` must not survive into this test."""
    log = logging.getLogger("root-logger-isolation")
    try:
        raise AttributeError("cannot unpack non-iterable coroutine object")
    except AttributeError:
        log.warning("quick-send resolve failed", exc_info=True)

    assert any(record.exc_info for record in caplog.records), [
        (record.getMessage(), record.exc_info, record.exc_text) for record in caplog.records
    ]

"""The opt-in a test uses when the NOTIFICATION PATH is its subject.

WHY THIS EXISTS, and why it is one module rather than an env poke per suite.
Two process-wide gates keep this repository's own tests away from the operator's
desktop — the kill switch ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` (armed at conftest
import, so spawned children inherit it) and the test-hosting rule
(``session.model_selection.session_uses_test_hosting``, which suppresses any
session recorded on the deterministic mock wire). A test whose SUBJECT is a
delivered toast or a ``notification`` frame has to clear both, and it has to do
it deliberately: without the first, nothing is composed; without the second, the
feed, the bridge and the TUI observer all decline the test-hosted session the
fixtures create, and the assertion has nothing to observe. That is exactly how
CI found the gap this module closes — the feed suites went red on the merge
because they had never been asked to opt in.

The escape itself is documented on its own constant in the product
(``model_selection.ENV_ALLOW_TEST_HOSTING_NOTIFY``), including the two
properties that make it safe: it can only ever ENABLE a notification, and the
kill switch still wins because every leg asks it first.

HOW TO USE IT, and the reason it is a context manager rather than a fixture.
Each module keeps its OWN opt-in fixture — that is this repository's established
style (see ``tests/unit/session/test_runtime_completion_announce.py`` and
``tests/e2e/test_desktop_sessions.py``), and a module-level fixture can depend on
the autouse fixture that arms the gates, which a shared one cannot do for both
the unit and the e2e trees. What is shared is the BODY:

    @pytest.fixture
    def notification_path_on() -> Iterator[None]:
        with notification_path_opt_in():
            yield

Set and restore are done in ``os.environ`` rather than through ``monkeypatch``
because the fixture is module-scoped in spirit and the switch is process-wide:
``monkeypatch`` is function-scoped and shared with the test, so a test calling
``monkeypatch.undo()`` would re-arm the gate mid-cell.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from local_operator.session.model_selection import ENV_ALLOW_TEST_HOSTING_NOTIFY
from local_operator.tui.notify import ENV_DISABLE

#: The two variables this module owns, in the order it sets them.
_OWNED = (ENV_DISABLE, ENV_ALLOW_TEST_HOSTING_NOTIFY)


@contextmanager
def notification_path_opt_in() -> Iterator[None]:
    """Clear the kill switch and waive the test-hosting rule, then restore both.

    The kill switch is REMOVED rather than set to a falsy value, because every
    reader tests truthiness and an empty string is the kind of value a later
    ``if`` can be wrong about. Whatever was there before is put back, so a run
    that legitimately silenced itself (a harness parent exporting the switch)
    stays silenced afterwards.
    """
    prior = {name: os.environ.get(name) for name in _OWNED}
    os.environ.pop(ENV_DISABLE, None)
    os.environ[ENV_ALLOW_TEST_HOSTING_NOTIFY] = "1"
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

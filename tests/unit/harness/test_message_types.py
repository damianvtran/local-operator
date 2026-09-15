"""Wire-format pins for the two markers nothing else pinned as literals.

``harness/message_types.py`` holds marker strings that are persisted into
transcripts and compared by equality across the session, the TUI, the phone
projection and a resumed replay, so changing a character of one orphans
existing rows instead of renaming anything. The module's docstring argues
that; a docstring cannot fail. These assertions are literals rather than
comparisons against the constant — a constant compared with itself passes
whatever the value becomes, which is the drift this file exists to catch.

Five of the seven markers are already pinned, but incidentally, by tests
asserting on something else (``test_incidents.py`` for the incident and
model-switch literals, ``test_wait_budget.py`` for the peer/hub arrival keys,
``test_session.py`` for the MCP-recovery type set, ``test_compose.py`` for
``resume``'s spelled-out copy). ``session_credential`` and ``todo_reminder``
were the two with no pin at all — exactly the pair the hoist that created this
module could have drifted with every suite still green. Pinning them closes
that gap, and keeps the seven covered by stated intent rather than by luck.
"""

from __future__ import annotations

from local_operator.harness.message_types import (
    SESSION_CREDENTIAL_MESSAGE_TYPE,
    TODO_REMINDER_MESSAGE_TYPE,
)


def test_unpinned_marker_values_are_stable() -> None:
    assert SESSION_CREDENTIAL_MESSAGE_TYPE == "session_credential"
    assert TODO_REMINDER_MESSAGE_TYPE == "todo_reminder"

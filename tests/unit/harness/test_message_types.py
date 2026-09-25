"""Wire-format pins for the markers nothing else pinned as literals.

``harness/message_types.py`` holds marker strings that are persisted into
transcripts and compared by equality across the session, the TUI, the phone
projection and a resumed replay, so changing a character of one orphans
existing rows instead of renaming anything. The module's docstring argues
that; a docstring cannot fail. These assertions are literals rather than
comparisons against the constant — a constant compared with itself passes
whatever the value becomes, which is the drift this file exists to catch.

Six of the nine markers are already pinned, but incidentally, by tests
asserting on something else (``test_incidents.py`` for the incident and
model-switch literals, ``test_wait_budget.py`` for the peer/hub arrival keys,
``test_session.py`` for the MCP-recovery type set, ``test_compose.py`` for
``resume``'s spelled-out copy). ``session_credential``, ``todo_reminder`` and
``session_credential_redaction`` were the three with no pin at all — the first
two are the pair the hoist that created this module could have drifted with
every suite still green, and the third joined them on 2026-09-24, when the
credential-shape notice moved off ``session_incident`` onto a type of its own.
That one needs the pin MORE than the other two rather than less: it is
PERSISTED, so it is read back by ``local-operator-ui``'s reducer and every
stored row is matched on the literal, and a drift would leave those rows
unrecognised with every suite green.
"""

from __future__ import annotations

from local_operator.harness.message_types import (
    SESSION_CREDENTIAL_MESSAGE_TYPE,
    SESSION_CREDENTIAL_REDACTION_MESSAGE_TYPE,
    TODO_REMINDER_MESSAGE_TYPE,
)


def test_unpinned_marker_values_are_stable() -> None:
    assert SESSION_CREDENTIAL_MESSAGE_TYPE == "session_credential"
    assert TODO_REMINDER_MESSAGE_TYPE == "todo_reminder"


def test_the_credential_redaction_marker_is_stable_and_is_not_an_incident() -> None:
    """The new persisted marker, plus the separation it was created for.

    The literal is the wire contract a stored row and the desktop reducer both
    match on. The INEQUALITY is the behavioural half of the same change: the
    record must not be an incident, because the incident type is exactly the one
    the renderer injects, and reusing it would put the notice back in front of
    the model. Asserted here as well as in the renderer's own test because a
    later "tidy up the two constants" would break the injection and only this
    file would say why not.
    """
    from local_operator.harness.message_types import SESSION_INCIDENT_MESSAGE_TYPE

    assert SESSION_CREDENTIAL_REDACTION_MESSAGE_TYPE == "session_credential_redaction"
    assert SESSION_CREDENTIAL_REDACTION_MESSAGE_TYPE != SESSION_INCIDENT_MESSAGE_TYPE
    assert SESSION_CREDENTIAL_REDACTION_MESSAGE_TYPE != SESSION_CREDENTIAL_MESSAGE_TYPE

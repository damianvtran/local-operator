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
PERSISTED, so it is read back by a consumer outside this repo —
``local-operator-ui``'s ``transcript-reducer.ts`` matches stored rows on the
literal — and a drift would orphan those rows with every suite here green.

What that consumer does with the type TODAY is measured, not hypothetical, and
it is accepted: the reducer's ``customRow()`` keys on ``"session_incident"`` and
the new type is not in its ``INLINE_CUSTOM_TYPES``, so a stored row takes the
generic ``relayRow(text)`` arm at ``level: "info"`` instead of ``incidentRow``
at ``level: "error"``. The row still renders its full text — a quieter
credential notice is the intended outcome — and the mismatch is recorded beside
the constant in ``harness/message_types.py`` so it is a decision rather than a
surprise. So the pin's job is narrower than "the reducer shows this row as an
incident": it fixes the WIRE VALUE, which is what a stored row is matched on,
and the inequality below fixes the separation the renderer's exclusion rests on.
Neither assertion here claims anything about the desktop ink.
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

    The literal is the wire contract a stored row is matched on — in this repo
    by the TUI's fold and the peek view, and outside it by ``local-operator-ui``'s
    transcript reducer, which keys stored rows on the literal and is NOT updated
    for this type (see the constant's note in ``harness/message_types.py``: the
    row still renders its text, at a quieter level, and that is accepted). The
    INEQUALITY is the behavioural half of the same change: the record must not be
    an incident, because the incident type is exactly the one the renderer
    injects, and reusing it would put the notice back in front of the model.
    Asserted here as well as in the renderer's own test because a later "tidy up
    the two constants" would break the injection and only this file would say why
    not.
    """
    from local_operator.harness.message_types import SESSION_INCIDENT_MESSAGE_TYPE

    assert SESSION_CREDENTIAL_REDACTION_MESSAGE_TYPE == "session_credential_redaction"
    assert SESSION_CREDENTIAL_REDACTION_MESSAGE_TYPE != SESSION_INCIDENT_MESSAGE_TYPE
    assert SESSION_CREDENTIAL_REDACTION_MESSAGE_TYPE != SESSION_CREDENTIAL_MESSAGE_TYPE

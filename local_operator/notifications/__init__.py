"""Notification CONTENT, composed once and rendered by every surface.

Deliberately not under ``tui/``. ``tui/notify.py``'s module docstring makes
"this module is imported only from ``tui/``" a load-bearing rule about
*delivery*: a backend that delivered its own toasts would double every alert
and would raise it on whichever machine the server happens to run on. That rule
is about delivery and it still holds — nothing here writes an escape sequence,
spawns ``notify-send`` or constructs an OS notification.

What was missing is the other half. The desktop app (local-operator-ui) owns
its own delivery surface but had no facts to render, so it hardcoded "Turn
complete / The agent finished its turn." while the session's name, its outcome
category and its last assistant line were all reachable in the backend and none
of them were on the wire. Composing here rather than in the renderer means one
place reads the ``display.notification_session_name`` privacy flag, one place
owns the wording, and a wording fix ships as a backend release rather than as a
signed Electron build.

See ``docs/design/descriptive-notifications.md`` for the decision record.
"""

from local_operator.notifications.compose import (
    NOTIFICATION_CONTRACT_VERSION,
    ComposedNotification,
    NotificationKind,
    compose,
    gate_body,
)

__all__ = [
    "NOTIFICATION_CONTRACT_VERSION",
    "ComposedNotification",
    "NotificationKind",
    "compose",
    "gate_body",
]

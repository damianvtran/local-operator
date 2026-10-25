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

THE PACKAGE INIT IS LAZY, AND THAT IS LOAD-BEARING. The package also holds the
operator's inbox (``store.py``, ``items.py``, ``presence.py``;
``docs/design/notifications-inbox.md``), whose store must stay stdlib-only: the
desktop feed's doorbell, the relay and the cleanup sweep read it where the
harness must not load. ``compose`` imports ``tui.notify``, which loads the TUI
package, pydantic and asyncio (~0.5 s, measured), so an eager re-export here
would put all of that on the path of every ``notifications.store`` import. The
composer's names are therefore resolved on first attribute access.

THE ``compose`` NAME IS GUARDED. The package exports the ``compose`` FUNCTION,
and the submodule is also called ``compose``. Python's import system binds a
freshly loaded submodule onto its parent package (``package.compose = <module>``)
the first time it is imported, so a caller whose first touch is
``from local_operator.notifications.compose import X`` would leave the package
attribute pointing at the MODULE, and ``notifications.compose(kind, ...)`` — the
seam the desktop bridge calls and T-B13's test patches — would raise
``TypeError: 'module' object is not callable``. The eager init used to hide this
by re-binding the function after the submodule loaded. :class:`_Package` keeps
that guarantee: a module assigned to the ``compose`` attribute is replaced by
its function. ``tests/unit/test_import_graph.py`` pins both import orders in a
fresh interpreter.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only; the runtime path is lazy
    from local_operator.notifications.compose import (
        FOCUS_ALWAYS,
        FOCUS_WHEN_UNFOCUSED,
        NOTIFICATION_CONTRACT_VERSION,
        ComposedNotification,
        NotificationKind,
        compose,
        gate_body,
        notification_payload,
    )

#: The package's public surface, unchanged from the eager version (which is
#: also what keeps the ``TYPE_CHECKING`` re-exports above from reading as
#: unused to flake8).
__all__ = [
    "FOCUS_ALWAYS",
    "FOCUS_WHEN_UNFOCUSED",
    "NOTIFICATION_CONTRACT_VERSION",
    "ComposedNotification",
    "NotificationKind",
    "compose",
    "gate_body",
    "notification_payload",
]

#: The composer's public names, resolved from ``compose.py`` on first access.
_COMPOSE_EXPORTS = tuple(__all__)

_COMPOSE_MODULE = f"{__name__}.compose"


class _Package(types.ModuleType):
    """This package, with the ``compose`` attribute kept a function.

    See the module docstring: the import system's parent binding would
    otherwise replace the exported function with the submodule of the same
    name. Only a MODULE assigned to ``compose`` is translated; any other value
    (a test's monkeypatched stand-in, the real function) is stored as given.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "compose" and isinstance(value, types.ModuleType):
            _bind_compose_exports(self, value)
            return
        super().__setattr__(name, value)


def _bind_compose_exports(package: types.ModuleType, module: types.ModuleType) -> None:
    for export in _COMPOSE_EXPORTS:
        types.ModuleType.__setattr__(package, export, getattr(module, export))


def __getattr__(name: str) -> Any:
    """Resolve a composer name on first access; see the module docstring."""
    if name in _COMPOSE_EXPORTS:
        import importlib

        module = importlib.import_module(_COMPOSE_MODULE)
        _bind_compose_exports(sys.modules[__name__], module)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _Package

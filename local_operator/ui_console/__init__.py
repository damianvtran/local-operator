"""The agent-facing half of the app's console surfaces (design §17.1 row C).

Three modules, each a sibling of the browser's own:

* :mod:`.state` — the FILE read that decides whether the ``console`` tool exists
  at all. The record is the app's, and the console rides it rather than owning a
  second discovery namespace (design §10.1).
* :mod:`.backend` — the authenticated loopback client. The transport is NOT
  re-implemented here: it is
  :class:`~local_operator.browser_bridge.backend.HostClient` pointed at this
  package's record, with the console's own failure copy and timeouts.
* :mod:`.policy` — the console-admission constants and the clamp, a clean leaf
  (``collections``/``typing`` only). ``config.py`` and ``settings_io.py`` read
  the three ``DEFAULT_*`` names at import, and the startup work requires those
  modules to stay free of asyncio/httpx/pydantic.

The console is one namespace on the app's EXISTING host — one key, one
heartbeat, one set of safety rules (bind ``127.0.0.1``, require
``X-Bridge-Key`` on every request, send no CORS headers, expose no CDP). It is
not a second endpoint, because a second listener would mean a second key, a
second heartbeat and a second set of those rules to keep true.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the names `__getattr__` serves, for type checkers only
    from local_operator.ui_console.backend import (  # noqa: F401
        CONSOLE_COPY,
        CONSOLE_METHODS,
        CONSOLE_TIMEOUTS,
        ConsoleHostClient,
        console_error_text,
        console_timeout,
        ui_console_advertisable,
        ui_console_available,
        ui_console_liveness,
    )
    from local_operator.ui_console.state import ConsoleHostState  # noqa: F401

#: Which submodule each re-exported name lives in, and why they are resolved
#: LAZILY rather than imported above.
#:
#: Importing ANY ``local_operator.ui_console.<sub>`` runs this ``__init__``
#: first, and the eager re-exports made ``.backend`` -- the httpx/asyncio/pydantic
#: transport -- the price of every such import. ``config.py`` and
#: ``settings_io.py`` import ``ui_console.policy`` (a clean leaf) at module
#: scope, so the eager ``.backend`` import dragged that whole stack back onto
#: ``import local_operator.cli`` -- exactly the ~40 modules the startup path
#: keeps off it, pinned by ``tests/unit/test_import_graph.py``.
#:
#: Nothing in this tree imports these names from the package; every caller names
#: the submodule. The re-exports are kept for compatibility, so PEP 562
#: ``__getattr__`` keeps ``from local_operator.ui_console import CONSOLE_COPY``
#: working exactly as before while deferring the cost to first access -- the same
#: pattern as ``local_operator/providers/__init__.py`` and
#: ``local_operator/mcp/__init__.py``. Do not "tidy" this back into
#: module-scope imports; the import-graph tests pin the startup graphs that
#: would regress.
_EXPORTS: dict[str, str] = {
    **dict.fromkeys(
        (
            "CONSOLE_COPY",
            "CONSOLE_METHODS",
            "CONSOLE_TIMEOUTS",
            "ConsoleHostClient",
            "console_error_text",
            "console_timeout",
            "ui_console_advertisable",
            "ui_console_available",
            "ui_console_liveness",
        ),
        "backend",
    ),
    "ConsoleHostState": "state",
}


def __getattr__(name: str) -> Any:
    """Resolve a deferred re-export on first access, then cache it in the module dict.

    Only names in :data:`_EXPORTS` are resolved here. A SUBMODULE reached as an
    attribute (``local_operator.ui_console.backend``) needs no help: once any
    code has imported it, the import system has already bound it on this
    package, and before that it was never reachable as an attribute either --
    the eager imports only bound the submodules they happened to name.
    """
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)

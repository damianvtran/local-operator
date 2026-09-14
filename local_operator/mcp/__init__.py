"""MCP support for the local-operator harness rewrite.

Official ``mcp`` Python SDK for transports + OAuth machinery; established
harness semantics for everything else: fast-startup gate with deferred tools, reconnect
circuit breaker, multi-source config discovery, tool-name mangling, and
outbound argument hygiene. See ``docs/REWRITE.md`` section E.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from local_operator.harness.types import AgentTool
from local_operator.mcp.tool_cache import McpToolCache

if TYPE_CHECKING:
    from local_operator.mcp.auth import ManagedAuthStore
    from local_operator.mcp.manager import McpLoadResult, McpManager

logger = logging.getLogger(__name__)

__all__ = [
    "McpLoadResult",
    "McpManager",
    "McpToolCache",
    "discover_and_load_mcp_tools",
]

#: The names this package re-exports from ``local_operator.mcp.manager``, and
#: the reason they are re-exported LAZILY rather than imported above.
#:
#: ``manager`` is a heavy module — it pulls ``local_operator.mcp.auth`` and
#: behind it the MCP SDK — and an eager import here made the whole manager
#: subtree the price of ANY ``local_operator.mcp.*`` import, including a pure
#: config read. ``session.frontend_state`` imports ``mcp.grants`` for one string
#: tuple, and ``session_factory._seed_mcp_routing`` imports ``mcp.config`` for
#: one JSON parse; both are on the runtime child's PRE-PUBLICATION path, where
#: every millisecond is in front of the user waiting for a bound session. The
#: manager itself is not wanted there at all — it is wanted when a turn actually
#: connects a server.
#:
#: PEP 562 module ``__getattr__`` keeps the names importable exactly as before
#: (``from local_operator.mcp import McpManager``, ``mcp.McpManager``) while
#: deferring the cost to the first attribute access. Do not "tidy" this back
#: into a module-scope import.
_MANAGER_EXPORTS = frozenset({"McpManager", "McpLoadResult"})


def __getattr__(name: str) -> Any:
    """Resolve the deferred exports, and keep module attribute access working.

    TWO THINGS BEYOND THE RE-EXPORT, both of which the eager import used to give
    away for free and neither of which has a caller in this tree — they are here
    so a future one does not meet a surprise:

    * ``local_operator.mcp.manager`` was an ATTRIBUTE of this package, because
      importing the name above imported the submodule. Reaching a submodule
      through its package is ordinary Python, so it still resolves here.
    * An ABSENT dependency must read as "no such attribute" rather than as an
      import error: ``getattr(local_operator.mcp, "McpManager", None)`` and
      ``hasattr`` are the shapes a capability probe uses, and the manager is an
      optional extra. A ``ModuleNotFoundError`` escaping a probe would turn
      "this machine has no MCP SDK" into a crash in the prober. The manager's
      own ``MCP_SDK_MISSING_ERROR`` path is unaffected — it is reached through
      ``local_operator.mcp.manager``, a real import that is expected to fail
      loudly where the SDK is genuinely required.

    The remaining accepted loss is documented rather than fixed:
    ``typing.get_type_hints`` on a function in this module cannot resolve a
    ``McpManager`` annotation, because the name is not in module globals until
    something touches the attribute. Nothing calls it; fixing it would mean
    keeping the eager import this block exists to remove.
    """
    if name == "manager":
        import importlib

        return importlib.import_module("local_operator.mcp.manager")
    if name in _MANAGER_EXPORTS:
        try:
            from local_operator.mcp.manager import McpLoadResult, McpManager
        except ImportError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

        # Cache into the module dict so the LOOKUP happens once per process
        # rather than once per access; ``__getattr__`` is only consulted for a
        # name the module does not already have.
        globals().update(McpLoadResult=McpLoadResult, McpManager=McpManager)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


async def discover_and_load_mcp_tools(
    cwd: str,
    tool_cache: McpToolCache | None = None,
    auth_store: ManagedAuthStore | None = None,
) -> tuple[McpManager, list[AgentTool], list[dict[str, str]]]:
    """Convenience loader that discovers and loads MCP tools in one pass.

    Returns ``(manager, tools, errors)``:

    - ``manager`` — the :class:`McpManager` (caller owns its lifecycle);
    - ``tools`` — the harness ``AgentTool`` list (live + deferred) sorted by
      name;
    - ``errors`` — ``[{"path": "mcp:<server>", "error": ...}]`` entries, one
      per failed or invalid server.

    A hard discovery failure never raises: it yields the manager, an empty
    tool list, and one synthetic error entry (established behavior).

    ``tool_cache`` defaults to :class:`McpToolCache` under :func:`config_dir`
    so a deferred server can advertise last-good schemas at the 250 ms gate
    without every runtime waiting on spawn+handshake. The owner still spawns;
    live ``tools/list`` after connect overwrites the row. Passing ``None``
    used to mean "no cache", which made every runtime pay the handshake even
    when a sibling had just listed the same server.

    The manager import is FUNCTION-LOCAL and goes through the PACKAGE attribute
    on purpose. Function-local because a module-scope name would be resolved
    from this module's globals, and a global lookup inside a function does not
    consult the package's PEP 562 ``__getattr__`` — only attribute access on the
    module object does. Through the package attribute so that
    ``local_operator.mcp.McpManager`` stays the single seam a caller (or a test)
    can substitute: reading the class straight off ``local_operator.mcp.manager``
    would quietly stop honouring a patch of the documented export.
    """
    from local_operator import mcp as mcp_package

    manager = mcp_package.McpManager(cwd, tool_cache or McpToolCache(), auth_store=auth_store)
    try:
        result = await manager.discover_and_connect()
    except Exception as exc:
        logger.warning("MCP discovery failed: %s", exc, exc_info=True)
        return manager, [], [{"path": ".mcp.json", "error": str(exc)}]

    errors = [{"path": f"mcp:{name}", "error": message} for name, message in result.errors.items()]
    return manager, result.tools, errors

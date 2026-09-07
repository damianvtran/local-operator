#!/usr/bin/env python3
"""Build the tool surface a REAL session advertises, for context measurement.

Why this exists
---------------
Every builder in ``local_operator.tools.registry`` follows the *createIf*
convention: it returns ``None`` when the capability its tool needs is absent
from the :class:`ToolContext`. That is correct for the product — a session
without a wake scheduler must not advertise ``wake`` — but it makes a bare
``ToolContext(cwd=...)`` a *misleading* thing to measure against. On this tree
a bare context builds 15 of 24 default tools; a fully-capable session builds
every one of them.

So any benchmark that constructs tools from a bare context is measuring a
surface no user ever has, and it understates the tool-schema cost — the single
largest per-request payload after the system prompt — by roughly a third. This
module exists so the budget guard measures the real thing.

How the gates are satisfied
---------------------------
The capability fields are typed as ``runtime_checkable`` Protocols (or bare
``Any``), and pydantic validates a Protocol-annotated field with an
``isinstance`` check. ``runtime_checkable`` tests only for the PRESENCE of the
protocol's attributes, never their signatures — but since 3.12 it resolves
them with ``inspect.getattr_static``, so a ``__getattr__`` catch-all does NOT
satisfy it and the attributes have to exist for real.

:func:`_stub_for` therefore synthesizes a class carrying exactly the names the
protocol declares (``__protocol_attrs__``), each bound to a no-op. Deriving
the names from the protocol rather than hardcoding them means a protocol that
grows a method does not silently drop a tool out of the measured surface —
which is the exact failure this module exists to correct.

The stubs are never CALLED: we build the tools to read their declared schemas
and throw them away.

This deliberately does NOT try to be a session. It is the smallest object that
makes the createIf gates say yes, so that the schemas we measure are the
schemas a real provider request would carry.
"""

from __future__ import annotations

from typing import Any

from local_operator.harness import types as _types
from local_operator.harness.types import AgentTool, ToolContext
from local_operator.tools import registry


def _stub_for(protocol: type, **extra: Any) -> Any:
    """An instance satisfying ``isinstance(obj, protocol)``.

    Attribute names come from the protocol itself, so this keeps working when
    a protocol gains a member. ``extra`` supplies the few attributes whose
    VALUE a builder inspects rather than merely requiring to exist.
    """
    names = set(getattr(protocol, "__protocol_attrs__", ()))
    namespace: dict[str, Any] = {
        name: (lambda self, *args, **kwargs: None) for name in names if name not in extra
    }
    namespace.update(extra)
    return type(f"_Stub{protocol.__name__}", (), namespace)()


class _UntypedStub:
    """Stand-in for the capability fields annotated ``Any`` (no isinstance).

    ``subagent_comms`` is the one whose value is read: ``hub``'s builder asks
    ``is_child(job_id)`` and returns the PARENT-side tool when it is false,
    which is the surface a fresh top-level session advertises.
    """

    def is_child(self, _job_id: Any = None) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def build_real_tool_context(cwd: str) -> ToolContext:
    """A ToolContext whose capabilities are all present.

    Mirrors what a fully-capable host injects, so ``create_tools`` takes the
    same branches it takes in a live session.
    """
    return ToolContext(
        cwd=cwd,
        # Each of these is a createIf gate; see the module docstring.
        wake_scheduler=_stub_for(_types.WakeSchedulerProtocol),
        subagent_launcher=_stub_for(_types.SubagentLauncher),
        jobs=_stub_for(_types.JobManagerProtocol),
        browser=_stub_for(_types.BrowserSurfaceProtocol, surface_id="stub"),
        subagent_comms=_UntypedStub(),
        agent_registry=_UntypedStub(),
        team_registry=_UntypedStub(),
        ask_user=lambda *args, **kwargs: None,  # pyright: ignore[reportArgumentType]
        # ``ask`` additionally requires an attached interactive surface.
        has_ui=True,
    )


def build_real_tools(cwd: str) -> list[AgentTool]:
    """The default tool set as a fully-capable session would advertise it."""
    return registry.create_tools(build_real_tool_context(cwd))


if __name__ == "__main__":  # pragma: no cover - developer convenience
    import os

    built = build_real_tools(os.getcwd())
    missing = sorted(set(registry.DEFAULT_TOOL_NAMES) - {t.name for t in built})
    print(f"built {len(built)} of {len(registry.DEFAULT_TOOL_NAMES)} default tools")
    print("  " + ", ".join(t.name for t in built))
    if missing:
        print(f"still gated off: {missing}")

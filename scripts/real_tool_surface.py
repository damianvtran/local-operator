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

from collections.abc import Iterator
from contextlib import contextmanager
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


async def _allow_quiet() -> None:
    """A quiet-end door that allows — the value a real session binds.

    ``build_no_reply_tool`` only checks PRESENCE, and every call the benchmark
    makes is a builder call, so the body is never run; it stays async to match
    the field's declared type.
    """
    return None


def build_real_tool_context(cwd: str) -> ToolContext:
    """A ToolContext whose capabilities are all present.

    Mirrors what a fully-capable host injects, so ``create_tools`` takes the
    same branches it takes in a live session.
    """
    return ToolContext(
        cwd=cwd,
        # Each of these is a createIf gate; see the module docstring.
        wake_scheduler=_stub_for(_types.WakeSchedulerProtocol),
        monitor_scheduler=_stub_for(_types.MonitorSchedulerProtocol),
        # The proactive CLASS is the patience tool's other createIf gate
        # (design §8.2.5): without it the measured surface is 30 of 31 tools —
        # the same "measures a surface no user has" defect the project
        # registry note below records (found by the context-budget job on the
        # class PR).
        action_class="proactive",
        subagent_launcher=_stub_for(_types.SubagentLauncher),
        jobs=_stub_for(_types.JobManagerProtocol),
        browser=_stub_for(_types.BrowserSurfaceProtocol, surface_id="stub"),
        subagent_comms=_UntypedStub(),
        agent_registry=_UntypedStub(),
        team_registry=_UntypedStub(),
        # The projects registry is the pair's createIf gate: without it
        # ``build_project_tool``/``build_project_delete_tool`` return ``None``
        # and the measured surface is 28 of 30 tools — the exact
        # "measures a surface no user has" defect this module exists to
        # prevent (found by the ``context-budget`` job on the projects PR).
        project_registry=_UntypedStub(),
        ask_user=lambda *args, **kwargs: None,  # pyright: ignore[reportArgumentType]
        # ``ask`` additionally requires an attached interactive surface.
        has_ui=True,
        # The agent-side settle (design §12): its builder is createIf-gated on
        # the callable the session binds while the queued engine is live — the
        # DEFAULT arm since the flip — so leaving it unbound here would measure
        # a surface no user has (the exact defect this module's docstring
        # records for the project registry and the patience class).
        withdraw_ask=lambda *args, **kwargs: {"ok": True},
        # The code-requests tool's createIf gate reads the session's own
        # DIRECTORY (its derived index lives beside the transcript) — context
        # DATA rather than a machine probe, so the stub simply carries one;
        # the tool is never called by a measurement. Without it the measured
        # surface is one tool lighter than a real session's, the same defect
        # the project registry above records.
        session_dir=f"{cwd}/.stub-session",
        # The quiet-end door (docs/design/quiet-turns.md §4): the newest
        # createIf gate, carried as a bare callable rather than a Protocol —
        # a real session binds it, so leaving it unbound here would measure a
        # surface one tool short of every session's (``no_reply``), the same
        # defect the project registry above records. The stub is never
        # awaited: the builder only checks presence.
        quiet_end=_allow_quiet,
    )


@contextmanager
def _forced_browser_backend() -> Iterator[None]:
    """Make ``build_browser_tool`` say yes regardless of the host.

    ``browser`` is the ONLY default tool whose gate ignores the ToolContext:
    it probes the machine (a cmux CLI on PATH, or the extension bridge's state
    files). Every other gate is satisfied by the stub context above, so this
    is the one capability a benchmark cannot express as data.

    That matters because it is measurement, not behaviour. Left unforced, this
    function returns 23 tools on a CI runner and 24 on a developer's cmux box
    — and ``browser`` is the single most expensive tool at 4,124 characters.
    A budget guard built on that reports 2,123 tokens of headroom on CI while
    a real host has 491, so the next tool added blows the budget on every real
    machine while CI stays green. That is precisely the green-by-fiction this
    benchmark was rewritten to eliminate, reintroduced one layer down.

    Patched into ``build_browser_tool.__globals__`` rather than by setting
    attributes on an imported module object. The two are normally the same
    dict, but only the former is guaranteed to be the namespace the function
    actually resolves its names from: a checkout that is BOTH on ``sys.path``
    and pip-installed can hold two distinct module objects for
    ``local_operator.tools.builtin``, and patching the copy this script
    imported then leaves the copy the builder reads untouched. That is not
    hypothetical — it is how this forcing silently did nothing on CI while
    working on the developer box, which the tool-count assertion in
    ``bench_context_budget`` caught.
    """
    from local_operator.tools.builtin import build_browser_tool

    namespace = build_browser_tool.__globals__
    names = ("cmux_browser_available", "bridge_browser_advertisable")
    saved = {n: namespace[n] for n in names}
    namespace.update({n: (lambda: True) for n in names})
    try:
        yield
    finally:
        namespace.update(saved)


@contextmanager
def _forced_console_backend() -> Iterator[None]:
    """Make ``build_console_tool`` say yes regardless of the host.

    The SECOND capability whose gate probes the machine rather than reading the
    :class:`ToolContext` (the browser above is the first): the console tool exists
    only where the desktop app publishes a console-capable record, so on a CI
    runner and on this fleet's hosts alike it would be gated off and the measured
    surface would be one tool lighter than a real session's. It is the same
    green-by-fiction the browser forcing exists to prevent, arriving through the
    same door: a budget guard whose answer depends on which machine ran it.

    Patched through ``__globals__`` for the reason :func:`_forced_browser_backend`
    spells out at length — a checkout both on ``sys.path`` and pip-installed can
    hold two distinct module objects, and patching the imported copy silently
    changes nothing in the namespace the builder reads.
    """
    from local_operator.tools.builtin import build_console_tool

    namespace = build_console_tool.__globals__
    name = "ui_console_advertisable"
    saved = namespace[name]
    namespace[name] = lambda: True
    try:
        yield
    finally:
        namespace[name] = saved


@contextmanager
def _forced_image_backend() -> Iterator[None]:
    """Make ``build_generate_image_tool`` say yes regardless of stored keys.

    The THIRD capability whose gate probes the machine rather than reading the
    ``ToolContext`` (browser, console, and now the image cascade): the tool
    exists only where some image provider credential is reachable, so on a CI
    runner — and on any host without a signed-in provider — it would be gated
    off and the measured surface one tool lighter than a session that HAS one.
    Same green-by-fiction the two above prevent, same fix.

    Patched on the module object the builder's globals actually hold
    (``image_availability``), so the attribute the builder resolves at call
    time is the one patched — the dual-module-copy hazard the browser leg
    documents at length is why the patch goes through ``__globals__``.
    """
    from local_operator.tools.image_tool import build_generate_image_tool

    namespace = build_generate_image_tool.__globals__
    availability = namespace["image_availability"]
    saved = availability.image_provider_reachable
    availability.image_provider_reachable = lambda *args, **kwargs: True
    try:
        yield
    finally:
        availability.image_provider_reachable = saved


def build_real_tools(cwd: str) -> list[AgentTool]:
    """The default tool set as a fully-capable session would advertise it.

    DETERMINISTIC across hosts: the count must not depend on whether the
    machine running the benchmark happens to have cmux, the desktop app, or an
    image provider credential. See :func:`_forced_browser_backend`,
    :func:`_forced_console_backend` and :func:`_forced_image_backend`. Callers
    that report a measurement should also report ``len()`` of this, so a drop
    below the full surface is visible rather than silent.
    """
    with (
        _forced_browser_backend(),
        _forced_console_backend(),
        _forced_image_backend(),
    ):
        return registry.create_tools(build_real_tool_context(cwd))


if __name__ == "__main__":  # pragma: no cover - developer convenience
    import os

    built = build_real_tools(os.getcwd())
    missing = sorted(set(registry.DEFAULT_TOOL_NAMES) - {t.name for t in built})
    print(f"built {len(built)} of {len(registry.DEFAULT_TOOL_NAMES)} default tools")
    print("  " + ", ".join(t.name for t in built))
    if missing:
        print(f"still gated off: {missing}")

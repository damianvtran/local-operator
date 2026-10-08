"""Deferred tool schemas: which built-in tools ship their schema on demand.

WHY THIS EXISTS. Every tool in a session's inventory used to put its full
schema in the provider ``tools`` array on every request. Seventeen tools that
under 2% of sessions ever call carried ~35k characters (~12k billed tokens) on
every call of every session (measured 2026-10-08 against ``analytics.db``; see
the PR that introduced this module). A deferred tool stays in the session's
INVENTORY — it resolves, validates, passes the approval gate and answers
``tool://`` exactly as before — and only its schema is left out of the array
the provider receives, until the session activates it.

WHAT THIS IS NOT. It is not a second gating convention beside ``createIf``
(``AGENTS.md``, "The tool-surface footprint ladder"): ``createIf`` decides
whether a tool EXISTS in a session; this decides only whether its schema rides
the request prefix. Enforcement (role allowlists, declared inventories, the
approval gate) is untouched because it reads the inventory, never the array.

Why the provider accepts a call to a tool absent from ``tools``: measured, not
assumed. One request per wire in active use (Anthropic, DeepSeek, OpenRouter,
Radient), each carrying a history with a call + result for a tool NOT in that
request's array, and each asked to call a tool it saw only in prompt text —
all accepted the history and all emitted the call. The table is on the PR.

A leaf module (it imports nothing from ``tools.builtin``) so the prompt
renderer can read the purpose phrases without pulling the tool builders in.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal

#: The session kinds that defer different sets. A child is a subagent; every
#: other session (TUI, desktop, exec, SDK) is ``top``.
DeferralKind = Literal["top", "child"]

#: Tools whose schema is withheld from a TOP-LEVEL session's tools array.
#:
#: Chosen from per-kind usage, not the blended figure: 93% of the 30-day
#: ledger's sessions are subagents, so a blended "share of sessions that called
#: it" hides that ``send`` is called by ~50% of top-level sessions and ~2% of
#: children. Every tool here is called by at most ~2% of top-level sessions;
#: ``lsp`` and ``patience`` by roughly none. ``ask`` is deliberately NOT here:
#: the ``<interactivity>`` bodies name it as the channel to the operator.
_TOP_LEVEL_DEFERRED: frozenset[str] = frozenset(
    {
        "console",
        "network",
        "team",
        "lsp",
        "patience",
        "ask_withdraw",
        "web_read",
        "project_delete",
        "team_delete",
        "read_variable",
        "list_variables",
    }
)

#: A child additionally defers the session-management tools a top-level
#: session uses heavily and a subagent almost never does (``send`` 2.1%,
#: ``secret`` 0.9%, ``agent`` 0.23%, ``project`` 0.17%, ``sessions`` 0.02% of
#: children). A role that NAMES one of them in its ``tools:`` list keeps it
#: published — see :func:`deferred_tool_names`.
_CHILD_DEFERRED: frozenset[str] = _TOP_LEVEL_DEFERRED | frozenset(
    {"project", "sessions", "send", "agent", "secret"}
)

DEFERRED_TOOLS: Mapping[DeferralKind, frozenset[str]] = {
    "top": _TOP_LEVEL_DEFERRED,
    "child": _CHILD_DEFERRED,
}

#: One short purpose phrase per deferrable tool, rendered in the inventory's
#: "schema on demand" line. The phrase is what lets a model decide to reach for
#: a tool whose description it has not been sent; keep each to a few words —
#: the full purpose is one ``read tool://<name>`` away. A deferrable tool with
#: no entry still renders (by name alone), so a missing phrase degrades rather
#: than hiding a tool; ``test_every_deferred_tool_has_a_purpose`` pins the map.
DEFERRED_TOOL_PURPOSES: Mapping[str, str] = {
    "console": "drive an interactive terminal",
    "network": "lop mesh peers and their sessions",
    "team": "author or list teams",
    "lsp": "code intelligence",
    "patience": "proactive reply-wait timers",
    "ask_withdraw": "settle or withdraw an open ask",
    "web_read": "answer from already-searched pages",
    "project_delete": "delete a project row",
    "team_delete": "delete a team",
    "read_variable": "read one variable",
    "list_variables": "list variable names",
    "project": "track multi-session workstreams",
    "sessions": "list/spawn/resume/stop lop sessions",
    "send": "message another lop session",
    "agent": "agent profiles and roles",
    "secret": "store and use credentials",
}

#: ``tools.defer`` — the kill switch. On by default; off publishes every
#: schema again (the pre-deferral behaviour) from the next turn.
TOOL_DEFERRAL_PATH: tuple[str, ...] = ("tools", "defer")
DEFAULT_TOOL_DEFERRAL = True


def tool_deferral_enabled(values: Mapping[str, object] | None = None) -> bool:
    """Read ``tools.defer``; anything but an explicit ``false`` means on.

    ``values`` is a config mapping already in hand (the config watcher's
    snapshot); ``None`` reads a fresh ``ConfigManager``. Never raises: a
    corrupt ``config.yml`` must not cost a session its tools array, and the
    default is the behaviour this ships as.
    """
    if values is None:
        try:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            section = ConfigManager(config_dir()).get_config_value(TOOL_DEFERRAL_PATH[0], None)
        except Exception:  # noqa: BLE001 — a config read never breaks a session
            return DEFAULT_TOOL_DEFERRAL
    else:
        section = values.get(TOOL_DEFERRAL_PATH[0]) if isinstance(values, Mapping) else None
    stored = section.get(TOOL_DEFERRAL_PATH[1]) if isinstance(section, Mapping) else None
    if isinstance(stored, bool):
        return stored
    return DEFAULT_TOOL_DEFERRAL


def deferred_tool_names(kind: DeferralKind, pinned: Collection[str] = ()) -> frozenset[str]:
    """The names ``kind`` defers, minus any a role/profile ``tools:`` list pins.

    A profile that NAMES a tool asked for it as part of what the role is, so
    withholding its schema would make the role's own core tool the one it
    has to discover. The lopdev ``manager`` seed names ``project``, for
    example, and ``read_variable``/``list_variables``.
    """
    return DEFERRED_TOOLS[kind] - frozenset(pinned)


def render_deferred_tools_line(names: Collection[str]) -> str:
    """The inventory's single line naming the tools whose schema is on demand.

    Fixed for a session's life — it lists the DEFERRABLE tools the session
    holds, not the ones still unpublished — so activating a tool does not move
    the inventory block and therefore costs no ``[session-state]`` delta.
    Ordered by name so the bytes do not depend on inventory order.
    """
    if not names:
        return ""
    entries = []
    for name in sorted(names):
        purpose = DEFERRED_TOOL_PURPOSES.get(name)
        entries.append(f"{name} ({purpose})" if purpose else name)
    return (
        "Schema on demand — call these directly by name, or read tool://<name> "
        "for the parameters: " + ", ".join(entries) + "."
    )


__all__ = [
    "DEFAULT_TOOL_DEFERRAL",
    "DEFERRED_TOOLS",
    "DEFERRED_TOOL_PURPOSES",
    "DeferralKind",
    "TOOL_DEFERRAL_PATH",
    "deferred_tool_names",
    "render_deferred_tools_line",
    "tool_deferral_enabled",
]

"""Session-scoped agent and team definitions (LOH-9).

WHY THIS EXISTS
===============

A role or team authored for ONE conversation must be usable by that conversation
— including its subagents, across turns, restarts and resumes — without entering
the user's global agent/team pickers. The global registry is both the delegation
authority and the picker vocabulary, so "ephemeral" cannot be a flag on a global
row: every reader would have to remember to filter it, and the first reader that
forgets leaks the row.

The scope is therefore a SEPARATE STORE on the same session filesystem:
``<session>/roles/agents`` and ``<session>/roles/teams``, in the same on-disk
formats as the global stores. A global reader that walks ``<config>/agents``
physically cannot see them; a session picker sees them because this module hands
the session a COMPOSITE that reads both stores.

S1 CONTRACT (merge-safety)
==========================

Writes delegate to the global store exactly as they did before this module
existed, so landing the read seam alone changes no behaviour. The composite does
route an ``update``/``delete``/``save`` to the store that OWNS the row id, which
is what lets a session row (placed by S2, or by a test) be edited without
touching an identically named global row. S2 moves ``create`` to the session
store and adds the explicit scope parameter at the call sites.

Identity is unchanged: ``AgentData`` and ``Team`` gain no field. Scope is
answered out-of-band by :meth:`ScopedAgentRegistry.scope_of` and the payloads'
additive ``scope`` key, so stored YAML, exports and mesh rows are untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

#: Directory under a session that holds its scoped definitions.
SESSION_ROLES_DIRNAME = "roles"

Scope = Literal["global", "session"]


def scope_dir_for(session_dir: str | Path) -> Path:
    """``<session>/roles`` — the root both scoped registries point at."""
    return Path(session_dir) / SESSION_ROLES_DIRNAME


class ScopedAgentRegistry:
    """Reads the session's agent store, then the global one; writes by owner.

    Duck-typed on purpose: every consumer already reaches a session's
    ``agent_registry`` through a small method surface (see
    ``local_operator.agent_profiles.resolve_profile`` and the ``agent`` tool).
    This class answers that surface and nothing else, so no caller has to know
    it is composite.
    """

    def __init__(
        self,
        global_registry: Any,
        session_registry: Any,
        *,
        scope_dir: Path,
    ) -> None:
        self.global_registry = global_registry
        self.session_registry = session_registry
        self.scope_dir = Path(scope_dir)

    # -- facts existing callers derive from a registry --------------------------
    @property
    def config_dir(self) -> Path:
        """The GLOBAL root. Paths derived from it must stay machine-wide."""
        return Path(self.global_registry.config_dir)

    @property
    def agents_dir(self) -> Path:
        return Path(self.global_registry.agents_dir)

    def scope_of(self, agent: Any) -> Scope:
        """Which store owns ``agent``. Session wins on an id it holds."""
        return "session" if self._session_row(getattr(agent, "id", "")) is not None else "global"

    def agent_dir(self, agent_id: str) -> Path:
        """The directory of ``agent_id`` in the store that owns it."""
        if self._session_row(agent_id) is not None:
            return Path(self.session_registry.agents_dir) / str(agent_id)
        return self.global_registry.agent_dir(agent_id)

    # -- reads ----------------------------------------------------------------
    def list_agents(self) -> list[Any]:
        return list(self.session_registry.list_agents()) + list(
            self.global_registry.list_agents()
        )

    def get_agent(self, agent_id: str) -> Any:
        row = self._session_row(agent_id)
        if row is not None:
            return row
        return self.global_registry.get_agent(agent_id)

    def get_agent_by_name(self, name: str) -> Any | None:
        row = self.session_registry.get_agent_by_name(name)
        if row is not None:
            return row
        return self.global_registry.get_agent_by_name(name)

    def get_agent_system_prompt(self, agent_id: str) -> str:
        if self._session_row(agent_id) is not None:
            return self.session_registry.get_agent_system_prompt(agent_id)
        return self.global_registry.get_agent_system_prompt(agent_id)

    def set_agent_system_prompt(self, agent_id: str, system_prompt: str) -> None:
        if self._session_row(agent_id) is not None:
            self.session_registry.set_agent_system_prompt(agent_id, system_prompt)
            return
        self.global_registry.set_agent_system_prompt(agent_id, system_prompt)

    def definition_revision_fields(self, agent_id: str) -> dict[str, Any]:
        if self._session_row(agent_id) is not None:
            return self.session_registry.definition_revision_fields(agent_id)
        return self.global_registry.definition_revision_fields(agent_id)

    def require_complete_metadata(self) -> None:
        """Strict on both stores; a scope that never wrote anything is empty.

        ``AgentRegistry.require_complete_metadata`` treats a MISSING agents dir
        as an unreadable registry. For a session scope that is the ordinary
        state of a conversation that never authored a role, so the session arm
        runs only when the directory exists; once it does, a broken row is
        strict like the global store.
        """
        self.global_registry.require_complete_metadata()
        if Path(self.session_registry.agents_dir).exists():
            self.session_registry.require_complete_metadata()

    def refresh_now(self) -> None:
        self.session_registry.refresh_now()
        self.global_registry.refresh_now()

    # -- writes (S1: global unless the row is already a session row) ----------
    def create_agent(self, fields: Any) -> Any:
        return self.global_registry.create_agent(fields)

    def save_agent(self, metadata: Any) -> Any:
        return self._owner_for_id(getattr(metadata, "id", "")).save_agent(metadata)

    def update_agent(
        self, agent_id: str, fields: Any, *, precondition: Any = None
    ) -> Any:
        return self._owner_for_id(agent_id).update_agent(
            agent_id, fields, precondition=precondition
        )

    def delete_agent(self, agent_id: str) -> None:
        self._owner_for_id(agent_id).delete_agent(agent_id)

    def clone_agent(self, agent_id: str, new_name: str) -> Any:
        return self._owner_for_id(agent_id).clone_agent(agent_id, new_name)

    # -- helpers --------------------------------------------------------------
    def _session_row(self, agent_id: str) -> Any | None:
        if not agent_id:
            return None
        try:
            return self.session_registry.get_agent(str(agent_id))
        except KeyError:
            return None

    def _owner_for_id(self, agent_id: str) -> Any:
        return (
            self.session_registry
            if self._session_row(str(agent_id)) is not None
            else self.global_registry
        )


class ScopedTeamRegistry:
    """The team twin of :class:`ScopedAgentRegistry`."""

    def __init__(
        self,
        global_registry: Any,
        session_registry: Any,
        *,
        scope_dir: Path,
    ) -> None:
        self.global_registry = global_registry
        self.session_registry = session_registry
        self.scope_dir = Path(scope_dir)

    @property
    def config_dir(self) -> Path:
        return Path(self.global_registry.config_dir)

    @property
    def teams_dir(self) -> Path:
        return Path(self.global_registry.teams_dir)

    def scope_of(self, team: Any) -> Scope:
        return "session" if self._session_row(getattr(team, "id", "")) is not None else "global"

    # -- reads ----------------------------------------------------------------
    def list_teams(self, *, recovery_wait: float | None = None) -> list[Any]:
        return list(self.session_registry.list_teams(recovery_wait=recovery_wait)) + list(
            self.global_registry.list_teams(recovery_wait=recovery_wait)
        )

    def get_team(self, team_id: str, *, recovery_wait: float | None = None) -> Any:
        row = self._session_row(team_id)
        if row is not None:
            return row
        return self.global_registry.get_team(team_id, recovery_wait=recovery_wait)

    def get_team_by_name(
        self, name: str, *, recovery_wait: float | None = None
    ) -> Any | None:
        row = self.session_registry.get_team_by_name(name, recovery_wait=recovery_wait)
        if row is not None:
            return row
        return self.global_registry.get_team_by_name(name, recovery_wait=recovery_wait)

    # -- writes (S1: global unless the row is already a session row) ----------
    def create_team(self, fields: Any) -> Any:
        return self.global_registry.create_team(fields)

    def save_team(self, team: Any) -> Any:
        return self._owner_for_id(getattr(team, "id", "")).save_team(team)

    def update_team(
        self, team_id: str, fields: Any, *, precondition: Any = None
    ) -> Any:
        return self._owner_for_id(team_id).update_team(
            team_id, fields, precondition=precondition
        )

    def delete_team(self, team_id: str) -> None:
        self._owner_for_id(team_id).delete_team(team_id)

    # -- helpers --------------------------------------------------------------
    def _session_row(self, team_id: str) -> Any | None:
        if not team_id:
            return None
        try:
            return self.session_registry.get_team(str(team_id))
        except KeyError:
            return None

    def _owner_for_id(self, team_id: str) -> Any:
        return (
            self.session_registry
            if self._session_row(str(team_id)) is not None
            else self.global_registry
        )


def scope_agent_registry(
    global_registry: Any, session_dir: str | Path
) -> Any:
    """Wrap ``global_registry`` with the session store in ``session_dir``.

    Idempotent: an already-scoped registry is returned unchanged, so a subagent
    that inherits its parent's registry cannot nest a second scope.
    """
    if global_registry is None:
        return None
    if getattr(global_registry, "scope_dir", None) is not None:
        return global_registry

    from local_operator.agents import AgentRegistry

    root = scope_dir_for(session_dir)
    # ``initialize=False``/``migrate=False``: reading a scope must not create
    # directories or run the global store's legacy migrations. ``save_agent``
    # makes the tree on the first real write.
    session_registry = AgentRegistry(root, initialize=False, migrate=False)
    return ScopedAgentRegistry(
        global_registry, session_registry, scope_dir=root
    )


def scope_team_registry(
    global_registry: Any, session_dir: str | Path
) -> Any:
    """The team twin of :func:`scope_agent_registry`."""
    if global_registry is None:
        return None
    if getattr(global_registry, "scope_dir", None) is not None:
        return global_registry

    from local_operator.teams import TeamRegistry

    root = scope_dir_for(session_dir)
    # TeamRegistry is already mkdir-free; a missing ``teams/`` is "no teams".
    session_registry = TeamRegistry(root)
    return ScopedTeamRegistry(
        global_registry, session_registry, scope_dir=root
    )


def scope_session_registries(
    agent_registry: Any,
    team_registry: Any,
    session_dir: str | Path,
) -> tuple[Any, Any]:
    """Both composites for one session, or the already-scoped pair unchanged."""
    if getattr(agent_registry, "scope_dir", None) is not None:
        return agent_registry, team_registry
    return (
        scope_agent_registry(agent_registry, session_dir),
        scope_team_registry(team_registry, session_dir),
    )


__all__ = [
    "SESSION_ROLES_DIRNAME",
    "ScopedAgentRegistry",
    "ScopedTeamRegistry",
    "scope_agent_registry",
    "scope_dir_for",
    "scope_session_registries",
    "scope_team_registry",
]

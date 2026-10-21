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
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # the host-facing consent type; importing it here would be heavy
    from local_operator.harness.types import DurableWriteIntent

#: Directory under a session that holds its scoped definitions.
SESSION_ROLES_DIRNAME = "roles"

Scope = Literal["global", "session"]

#: Every scope, in the order a message names them.
SCOPES: tuple[Scope, ...] = ("session", "global")


def scope_dir_for(session_dir: str | Path) -> Path:
    """``<session>/roles`` — the root both scoped registries point at."""
    return Path(session_dir) / SESSION_ROLES_DIRNAME


def scope_words(scope: Scope) -> str:
    """The PLAIN WORDS a receipt or a refusal uses for one store.

    Receipts say where a write landed, so the two scopes must never be named by
    their identifiers: a user reading a transcript sees "in your account" or
    "in this conversation", never ``global``/``session`` (design §3.5).
    """
    return "your account" if scope == "global" else "this conversation"


def row_scope(registry: Any, row: Any) -> Scope:
    """Which store owns ``row``. A registry with no scopes owns everything.

    The fallback is ``global`` on purpose: a plain ``AgentRegistry`` is the
    account's store, and treating an unknown owner as session-scoped would let
    a tool delete a durable row through a host whose registry it cannot
    inspect. Only the composite, which can answer ``scope_of``, may say
    ``session``.
    """
    scope_of = getattr(registry, "scope_of", None)
    return scope_of(row) if callable(scope_of) else "global"


def scope_store(registry: Any, scope: Scope) -> Any:
    """The store a write at ``scope`` must go through. NEVER a silent default.

    Both write entry points (``tools.agent_tool.write_profile`` and
    ``tools.team_tool._op_write``) call this with the scope the CALLER chose, so
    a missed adapter is a visible refusal rather than a write that lands in the
    wrong store. On a composite it returns that scope's write adapter (which
    takes the composite's create lock); on a plain registry ``global`` returns
    the registry itself.

    ``session`` on a plain registry is REFUSED, not downgraded to global: a host
    with no persistent session directory has no conversation scope (design
    §3.2), and writing the row account-wide while the caller asked for
    conversation-only is the exact leak this module exists to prevent.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; use 'session' or 'global'.")
    store_for = getattr(registry, "store_for", None)
    if callable(store_for):
        return store_for(scope)
    if scope == "global":
        return registry
    raise ValueError(
        "this run has no conversation store, so a role or team cannot be kept for "
        "this conversation only. Save it for every conversation instead "
        "(scope='global', which needs the user's confirmation), or create it in "
        "the desktop Profiles editor, the TUI /agent or /team form, or `lop agents` "
        "or `lop teams`."
    )


def _shadow_refusal(kind: str, name: str) -> str:
    """Refuse a conversation-only row whose name already exists in the account.

    A session row SHADOWS the account's row with the same name, so allowing this
    create would silently change what `task(agent=<name>)` or `/team <name>`
    launches in this conversation — the reuse-first ladder inverted. The caller
    is told both ways forward: use the row that already exists, or pick a name
    that says what makes this version different.
    """
    launch = f"/team {name}" if kind == "team" else f"task(agent={name!r})"
    return (
        f"a {kind} named {name!r} already exists in your account, so this conversation "
        f"already has it: launch it as it is ({launch}). Choose a different name for "
        "the conversation-only version; copying one into your account is a separate step."
    )


async def durable_write_origin(context: Any, intent: "DurableWriteIntent") -> str | None:
    """Ask the HOST to approve a write into the account's shared store.

    THE ONE CONSENT SEAM (LOH-9 S2). The decision is the host's: a model must not
    be able to stamp its own provenance, so there is no grant field on any tool
    parameter and no value in this module authorizes anything. ``None`` means
    "no seam here" or "the user said no" and is a REFUSAL; a non-empty string is
    the host's audit origin for a write it approved, and is recorded, never
    re-read as authority.

    A seam that raises is also a refusal: an ask that could not be delivered is
    not a yes, and failing open here would let a broken host silently grant
    account-wide writes.
    """
    callback = getattr(context, "authorize_durable_write", None) if context is not None else None
    if callback is None:
        return None
    try:
        origin = await callback(intent)
    except Exception:  # noqa: BLE001 — an unavailable consent seam is a refusal
        logger.warning("durable-write consent seam failed; refusing", exc_info=True)
        return None
    return str(origin) if origin else None


def durable_write_refusal(intent: "DurableWriteIntent") -> str:
    """Why a tool-issued account write was refused, and the two ways forward.

    Names the user surfaces that ARE the user's own act (desktop editor, TUI form,
    CLI) and the retry path that needs no consent at all (the conversation-only
    default), because a refusal that does not say what to do next is how a model
    starts guessing at scope values.
    """
    kind = intent.kind
    if kind == "team":
        surface = "the desktop Teams editor, the TUI /team form, or `lop teams`"
    else:
        surface = "the desktop Profiles editor, the TUI /agent form, or `lop agents`"
    return (
        f"saving a {kind} for every conversation needs the user's confirmation, and "
        "this run cannot ask for it. To keep it for THIS conversation only, retry "
        "without scope (or with scope='session'). To add it to the user's account "
        f"yourself, use {surface}."
    )


class _ScopedAgentWrites:
    """The agent-registry write surface over ONE store.

    Reads are answered by the store it wraps, NOT by the composite on purpose: a
    create's name check has to see the rows of the store the new row will sit in.
    The composite's reads are the picker's view; this is the writer's view, and a
    global-scope write must not be refused because a conversation row happens to
    share the name.

    Writes take the OWNER's ``write_lock``, so a create's uniqueness check and the
    row it writes are one critical section, across both stores (see
    ``ScopedAgentRegistry.__init__``).
    """

    def __init__(self, owner: "ScopedAgentRegistry", scope: Scope) -> None:
        self._owner = owner
        self.scope: Scope = scope
        self._store = owner.session_registry if scope == "session" else owner.global_registry

    # -- reads: the wrapped store only ---------------------------------------
    def list_agents(self) -> list[Any]:
        return self._store.list_agents()

    def get_agent(self, agent_id: str) -> Any:
        return self._store.get_agent(agent_id)

    def get_agent_by_name(self, name: str) -> Any | None:
        return self._store.get_agent_by_name(name)

    def get_agent_system_prompt(self, agent_id: str) -> str:
        return self._store.get_agent_system_prompt(agent_id)

    def definition_revision_fields(self, agent_id: str) -> dict[str, Any]:
        return self._store.definition_revision_fields(agent_id)

    # -- writes --------------------------------------------------------------
    def set_agent_system_prompt(self, agent_id: str, system_prompt: str) -> None:
        with self._owner.write_lock:
            self._store.set_agent_system_prompt(agent_id, system_prompt)

    def create_agent(self, fields: Any) -> Any:
        with self._owner.write_lock:
            if self.scope == "session":
                # A conversation row SHADOWS the account's row of the same name,
                # so this create would silently change what `task(agent=<name>)`
                # launches here. Refuse and say both ways forward (design §3.2
                # "name collisions"); S4's promotion is the copy-in door, not a
                # create that looks like one.
                name = str(getattr(fields, "name", "") or "").strip()
                shadow = self._owner.global_registry.get_agent_by_name(name)
                if shadow is not None:
                    raise ValueError(_shadow_refusal("agent", name))
            return self._store.create_agent(fields)

    def save_agent(self, metadata: Any) -> Any:
        with self._owner.write_lock:
            return self._store.save_agent(metadata)

    def update_agent(self, agent_id: str, fields: Any, *, precondition: Any = None) -> Any:
        with self._owner.write_lock:
            return self._store.update_agent(agent_id, fields, precondition=precondition)

    def delete_agent(self, agent_id: str) -> None:
        with self._owner.write_lock:
            self._store.delete_agent(agent_id)

    def clone_agent(self, agent_id: str, new_name: str) -> Any:
        with self._owner.write_lock:
            return self._store.clone_agent(agent_id, new_name)


class _ScopedTeamWrites:
    """The team twin of :class:`_ScopedAgentWrites`."""

    def __init__(self, owner: "ScopedTeamRegistry", scope: Scope) -> None:
        self._owner = owner
        self.scope: Scope = scope
        self._store = owner.session_registry if scope == "session" else owner.global_registry

    def list_teams(self, *, recovery_wait: float | None = None) -> list[Any]:
        return self._store.list_teams(recovery_wait=recovery_wait)

    def get_team(self, team_id: str, *, recovery_wait: float | None = None) -> Any:
        return self._store.get_team(team_id, recovery_wait=recovery_wait)

    def get_team_by_name(self, name: str, *, recovery_wait: float | None = None) -> Any | None:
        return self._store.get_team_by_name(name, recovery_wait=recovery_wait)

    def create_team(self, fields: Any) -> Any:
        with self._owner.write_lock:
            if self.scope == "session":
                name = str(getattr(fields, "name", "") or "").strip()
                shadow = self._owner.global_registry.get_team_by_name(name)
                if shadow is not None:
                    raise ValueError(_shadow_refusal("team", name))
            return self._store.create_team(fields)

    def save_team(self, team: Any) -> Any:
        with self._owner.write_lock:
            return self._store.save_team(team)

    def update_team(self, team_id: str, fields: Any, *, precondition: Any = None) -> Any:
        with self._owner.write_lock:
            return self._store.update_team(team_id, fields, precondition=precondition)

    def delete_team(self, team_id: str) -> None:
        with self._owner.write_lock:
            self._store.delete_team(team_id)


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
        # ONE create lock for BOTH scopes. Two subagents creating the same name
        # must not both pass the uniqueness check: the stores have their own
        # locks, but a session create also checks the GLOBAL store for a shadow,
        # and that check plus the write is one critical section across stores.
        self._write_lock = threading.RLock()
        self._stores: dict[Scope, _ScopedAgentWrites] = {
            scope: _ScopedAgentWrites(self, scope) for scope in SCOPES
        }

    def store_for(self, scope: Scope) -> "_ScopedAgentWrites":
        """The write adapter for ``scope`` — see :func:`scope_store`."""
        if scope not in SCOPES:
            raise ValueError(f"unknown scope {scope!r}; use 'session' or 'global'.")
        return self._stores[scope]

    @property
    def write_lock(self) -> Any:
        """The cross-store lock every write adapter takes around a mutation."""
        return self._write_lock

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
        return list(self.session_registry.list_agents()) + list(self.global_registry.list_agents())

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

    # -- writes: the SCOPE IS THE CALLER'S, never a default in here -----------
    # A default here would be the silent leak this module exists to prevent: a
    # caller that forgot to choose would write account-wide and look correct.
    def create_agent(self, fields: Any, *, scope: Scope) -> Any:
        return self.store_for(scope).create_agent(fields)

    def save_agent(self, metadata: Any) -> Any:
        with self._write_lock:
            return self._owner_for_id(getattr(metadata, "id", "")).save_agent(metadata)

    def update_agent(self, agent_id: str, fields: Any, *, precondition: Any = None) -> Any:
        with self._write_lock:
            return self._owner_for_id(agent_id).update_agent(
                agent_id, fields, precondition=precondition
            )

    def delete_agent(self, agent_id: str) -> None:
        with self._write_lock:
            self._owner_for_id(agent_id).delete_agent(agent_id)

    def clone_agent(self, agent_id: str, new_name: str) -> Any:
        with self._write_lock:
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
        # See ``ScopedAgentRegistry.__init__``: one lock across both stores,
        # because a session create checks the global store for a shadow.
        self._write_lock = threading.RLock()
        self._stores: dict[Scope, _ScopedTeamWrites] = {
            scope: _ScopedTeamWrites(self, scope) for scope in SCOPES
        }

    def store_for(self, scope: Scope) -> "_ScopedTeamWrites":
        """The write adapter for ``scope`` — see :func:`scope_store`."""
        if scope not in SCOPES:
            raise ValueError(f"unknown scope {scope!r}; use 'session' or 'global'.")
        return self._stores[scope]

    @property
    def write_lock(self) -> Any:
        """The cross-store lock every write adapter takes around a mutation."""
        return self._write_lock

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

    def get_team_by_name(self, name: str, *, recovery_wait: float | None = None) -> Any | None:
        row = self.session_registry.get_team_by_name(name, recovery_wait=recovery_wait)
        if row is not None:
            return row
        return self.global_registry.get_team_by_name(name, recovery_wait=recovery_wait)

    # -- writes: the SCOPE IS THE CALLER'S, never a default in here -----------
    def create_team(self, fields: Any, *, scope: Scope) -> Any:
        return self.store_for(scope).create_team(fields)

    def save_team(self, team: Any) -> Any:
        with self._write_lock:
            return self._owner_for_id(getattr(team, "id", "")).save_team(team)

    def update_team(self, team_id: str, fields: Any, *, precondition: Any = None) -> Any:
        with self._write_lock:
            return self._owner_for_id(team_id).update_team(
                team_id, fields, precondition=precondition
            )

    def delete_team(self, team_id: str) -> None:
        with self._write_lock:
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


def scope_agent_registry(global_registry: Any, session_dir: str | Path) -> Any:
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
    return ScopedAgentRegistry(global_registry, session_registry, scope_dir=root)


def scope_team_registry(global_registry: Any, session_dir: str | Path) -> Any:
    """The team twin of :func:`scope_agent_registry`."""
    if global_registry is None:
        return None
    if getattr(global_registry, "scope_dir", None) is not None:
        return global_registry

    from local_operator.teams import TeamRegistry

    root = scope_dir_for(session_dir)
    # TeamRegistry is already mkdir-free; a missing ``teams/`` is "no teams".
    session_registry = TeamRegistry(root)
    return ScopedTeamRegistry(global_registry, session_registry, scope_dir=root)


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

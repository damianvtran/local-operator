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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    scope_of: Callable[[Any], Scope] | None = getattr(registry, "scope_of", None)
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


def tool_write_scope(registry: Any, requested: "Scope | None") -> Scope:
    """The store a TOOL write lands in when the caller named none.

    A host WITH a conversation store defaults to ``session`` (design §3.4: the
    agent-authored default is ephemeral). A host WITHOUT one — an embedder, a
    bare ``--agent`` run, a test rig with a plain global registry — keeps the
    behaviour it had before this slice: the omission writes the ACCOUNT store,
    because there is no conversation scope to default into. That is not a
    silent downgrade of a request; nothing was requested. An EXPLICIT
    ``scope='session'`` on such a host is still refused by :func:`scope_store`.
    """
    if requested is not None:
        return requested
    return "session" if getattr(registry, "session_registry", None) is not None else "global"


def needs_durable_consent(registry: Any, requested: "Scope | None") -> bool:
    """Whether this tool call must ask the host before it mutates the account.

    Two cases, both model-initiated account writes: an EXPLICIT ``scope='global'``
    on any host, and any write that lands on an account row of a SCOPED host
    (create defaulting global cannot happen there, but update/delete of an
    existing account row can). A host with NO conversation store has no seam and
    no alternative store — its tool writes are the pre-S2 behaviour and are not
    gated, or every role-management tool on an embedder would become read-only.
    """
    if requested is not None:
        return True
    return getattr(registry, "session_registry", None) is not None


#: How long a role/team write waits for the cross-store lock before refusing.
#: A create checks the OTHER store and then writes, so the two steps must be one
#: critical section; a holder that dies or blocks forever must not freeze the
#: tool that is waiting (the team tool has the same bound for the same reason).
WRITE_LOCK_TIMEOUT_S = 10.0


@contextmanager
def _write_section(owner: Any) -> Iterator[None]:
    """Take the composite's write lock with a BOUND, or refuse in words.

    The tool entry points call the registry synchronously from an async tool
    body, so an unbounded wait would block the session's event loop with no
    output. A timeout is a clear refusal the caller can act on ("retry"), and the
    measured contention this lock exists for (two subagents creating one name) is
    a sub-millisecond critical section, so a 10 s wait is already pathological.
    """
    lock = owner.write_lock
    if not lock.acquire(timeout=WRITE_LOCK_TIMEOUT_S):
        raise TimeoutError(
            "timed out waiting for the role write lock, so nothing was written; "
            "another write is still running — retry in a moment"
        )
    try:
        yield
    finally:
        lock.release()


def shadow_refusal(kind: str, name: str, *, existing_scope: Scope) -> str:
    """Refuse a create whose name already resolves in the OTHER store.

    A session row SHADOWS the account's row with the same name, so allowing one
    would silently change what `task(agent=<name>)` or `/team <name>` launches in
    this conversation — the reuse-first ladder inverted. The reverse is just as
    confusing: an account row created while this conversation already has one is
    unreachable HERE. Each case names the one action that resolves it.
    """
    if existing_scope == "global":
        launch = f"/team {name}" if kind == "team" else f"task(agent={name!r})"
        return (
            f"a {kind} named {name!r} already exists in your account, and this "
            f"conversation uses it: launch it as it is ({launch}). Choose a different "
            "name for the conversation-only version."
        )
    delete = "team_delete" if kind == "team" else "the agent tool's op='delete'"
    return (
        f"this conversation already has a {kind} named {name!r}, so an account row of "
        "that name would be hidden here. Delete the conversation one first "
        f"({delete}), or choose another name."
    )


def shadow_in_other_store(registry: Any, scope: Scope, name: str, *, kind: str) -> Any | None:
    """The row in the OTHER store that ``name`` would collide with, or ``None``.

    Case-insensitive, and for teams ALIASES count: ``/team release`` resolves an
    alias, so a session team named "release" shadows an account team that answers
    to that key exactly as a name collision would. A plain registry has one store,
    so nothing can collide across scopes.
    """
    other = getattr(registry, "session_registry" if scope == "global" else "global_registry", None)
    if other is None:
        return None
    wanted = name.strip().casefold()
    if not wanted:
        return None
    if kind == "team":
        for team in other.list_teams():
            keys = [str(getattr(team, "name", "") or "")]
            keys.extend(str(alias) for alias in (getattr(team, "aliases", ()) or ()))
            if any(key.strip().casefold() == wanted for key in keys if key):
                return team
        return None
    from local_operator.agent_profiles import is_role, is_specialist

    for agent in other.list_agents():
        # ONLY the rows a create would shadow as a ROLE. An ordinary
        # conversational agent of the same name is refused by ``write_profile``'s
        # own "not a role" sentence, which is the more useful answer (it says the
        # name is taken by something that is not a role at all).
        if not (is_role(agent) or is_specialist(agent)):
            continue
        if str(getattr(agent, "name", "") or "").strip().casefold() == wanted:
            return agent
    return None


async def durable_write_decision(
    context: Any, intent: "DurableWriteIntent"
) -> tuple[str | None, str]:
    """Ask the HOST to approve a write into the account's shared store.

    THE ONE CONSENT SEAM (LOH-9 S2). The decision is the host's: a model must not
    be able to stamp its own provenance, so there is no grant field on any tool
    parameter and no value in this module authorizes anything. A non-empty string
    is the host's AUDIT origin for a write it approved, recorded and never re-read
    as authority. A seam that raises is a refusal: an ask that could not be
    delivered is not a yes, and failing open here would let a broken host silently
    grant account-wide writes.

    Returns ``(origin, reason)`` where reason is ``granted``, ``no_seam`` (this
    host has no approval channel — desktop/headless today), ``declined`` (the
    callback answered None: the user said no) or ``error`` (the channel failed).
    The distinction is the refusal's subject: "this run cannot ask" and "the user
    declined" are different facts, and conflating them told the model to re-ask a
    question the user had already answered.
    """
    callback = getattr(context, "authorize_durable_write", None) if context is not None else None
    if callback is None:
        return None, "no_seam"
    try:
        origin = await callback(intent)
    except Exception:  # noqa: BLE001 — an unavailable consent seam is a refusal
        logger.warning("durable-write consent seam failed; refusing", exc_info=True)
        return None, "error"
    return (str(origin), "granted") if origin else (None, "declined")


async def durable_write_origin(context: Any, intent: "DurableWriteIntent") -> str | None:
    """The origin-only form of :func:`durable_write_decision` (existing callers)."""
    origin, _reason = await durable_write_decision(context, intent)
    return origin


def durable_write_refusal(intent: "DurableWriteIntent", reason: str = "no_seam") -> str:
    """Why a tool-issued account write was refused, and the ONE next action.

    The reason is named rather than folded into "cannot ask": a host without an
    approval channel and a user who declined are different facts, and the second
    must not invite a retry of the same question. The remedy is ONE concrete
    surface for this intent — a menu of three vague ones made the model pick.
    The conversation-only retry stays, because it is the path that needs no
    consent at all and it is what the scope default does.
    """
    article = "an" if intent.kind == "agent" else "a"
    if reason == "declined":
        opening = (
            f"the user declined to save this {intent.kind} for every conversation, so "
            "nothing in their account was changed."
            if intent.operation != "sync"
            else "the user declined the account updates, so no role saved in their "
            "account was changed."
        )
    elif reason == "error":
        opening = (
            "the confirmation channel failed before the user could answer, so nothing "
            "in their account was changed."
        )
    else:
        opening = (
            f"saving {article} {intent.kind} for every conversation needs the user's "
            "confirmation, and this run has no way to ask."
        )
    if intent.kind == "team":
        action = "the TUI `/team` form"
    elif intent.operation == "sync":
        action = "`lop agents sync`"
    else:
        action = "the TUI `/agent` form"
    return (
        f"{opening} To keep it for THIS conversation only, retry without scope (or with "
        f"scope='session'). To change the user's account, ask them to use {action}."
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
        with _write_section(self._owner):
            self._store.set_agent_system_prompt(agent_id, system_prompt)

    def create_agent(self, fields: Any) -> Any:
        with _write_section(self._owner):
            if self.scope == "session":
                # A conversation row SHADOWS the account's row of the same name,
                # so this create would silently change what `task(agent=<name>)`
                # launches here. Refuse and say both ways forward (design §3.2
                # "name collisions"); S4's promotion is the copy-in door, not a
                # create that looks like one.
                name = str(getattr(fields, "name", "") or "").strip()
                shadow = self._owner.global_registry.get_agent_by_name(name)
                if shadow is not None:
                    raise ValueError(shadow_refusal("agent", name, existing_scope="global"))
            return self._store.create_agent(fields)

    def save_agent(self, metadata: Any) -> Any:
        with _write_section(self._owner):
            return self._store.save_agent(metadata)

    def update_agent(self, agent_id: str, fields: Any, *, precondition: Any = None) -> Any:
        with _write_section(self._owner):
            return self._store.update_agent(agent_id, fields, precondition=precondition)

    def delete_agent(self, agent_id: str) -> None:
        with _write_section(self._owner):
            self._store.delete_agent(agent_id)

    def clone_agent(self, agent_id: str, new_name: str) -> Any:
        with _write_section(self._owner):
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
        with _write_section(self._owner):
            if self.scope == "session":
                name = str(getattr(fields, "name", "") or "").strip()
                shadow = self._owner.global_registry.get_team_by_name(name)
                if shadow is not None:
                    raise ValueError(shadow_refusal("team", name, existing_scope="global"))
            return self._store.create_team(fields)

    def save_team(self, team: Any) -> Any:
        with _write_section(self._owner):
            return self._store.save_team(team)

    def update_team(self, team_id: str, fields: Any, *, precondition: Any = None) -> Any:
        with _write_section(self._owner):
            return self._store.update_team(team_id, fields, precondition=precondition)

    def delete_team(self, team_id: str) -> None:
        with _write_section(self._owner):
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

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
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from local_operator import role_provenance

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
    article = "an" if kind == "agent" else "a"
    if existing_scope == "global":
        launch = f"/team {name}" if kind == "team" else f"task(agent={name!r})"
        return (
            f"{article} {kind} named {name!r} already exists in your account, and this "
            f"conversation uses it: launch it as it is ({launch}). Choose a different "
            "name for the conversation-only version."
        )
    delete = "team_delete" if kind == "team" else "the agent tool's op='delete'"
    return (
        f"this conversation already has {article} {kind} named {name!r}, so an account row of "
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

    The conversation-only retry is offered ONLY where it can succeed (round 3,
    U7): create and install choose their store, so retrying without scope lands
    in this conversation. Update and reset act on an EXISTING row, whose store
    the retry does not change — offering it there reproduced the same refusal
    byte for byte. Those two name the user-surface action instead. ``sync`` is
    account-only and has no conversation retry either.

    ``promote`` (S4) has no retry of ANY kind: the definition is already in this
    conversation, and what it asks for is a second copy elsewhere. Its remedy is
    the two surfaces that have the user in front of them — the desktop action and
    the TUI verb — so the refusal never implies the agent can do it alone.
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
    elif intent.operation == "promote":
        opening = (
            f"copying this {intent.kind} into the user's account is their decision, and "
            "this run has no way to ask."
        )
    else:
        opening = (
            f"saving {article} {intent.kind} for every conversation needs the user's "
            "confirmation, and this run has no way to ask."
        )
    if intent.operation == "promote":
        # The copy already exists HERE; what is missing is the user's own act on a
        # surface that can show them exactly what will be copied.
        noun = "/agent promote" if intent.kind == "agent" else "/team promote"
        return (
            f"{opening} Ask them to use the desktop's Use-in-all-conversations action "
            f"or the TUI `{noun} {intent.name}` verb: both name exactly what will be "
            "copied, and neither can be run on their behalf."
        )
    if intent.kind == "team":
        action = "the TUI `/team` form"
    elif intent.operation == "sync":
        action = "`lop agents sync`"
    else:
        action = "the TUI `/agent` form"
    if intent.operation in ("create", "install"):
        return (
            f"{opening} To keep it for THIS conversation only, retry without scope (or with "
            f"scope='session'). To change the user's account, ask them to use {action}."
        )
    return f"{opening} To change the user's account, ask them to use {action}."


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

    # -- promotion: copy THIS conversation's role into the account (LOH-9 S4) --
    def promote(
        self,
        name: str,
        *,
        as_name: str | None = None,
        replace: bool = False,
        authorized_by: str | None = None,
        dry_run: bool = False,
    ) -> PromoteResult:
        """Copy the conversation role ``name`` into the account store.

        COPY, NOT MOVE: the conversation row is untouched, so this conversation
        keeps launching its own row (session-first resolution) while every other
        conversation gains the account copy. That is deliberate and the receipt
        says it, because a user who promotes and then chats here still sees the
        original — the copy is for everywhere ELSE.

        ``replace`` is the caller's confirmation that an existing account row of
        the target name may be overwritten. There is no separate confirmation
        flag here: the human answer lives on the surface that has one (the
        desktop body's ``confirm_replace``, the TUI's typed ``yes``, the tool's
        ``confirm_replace``), and faking it at this layer is exactly what the
        consent rule forbids.

        ``dry_run`` runs the SAME preflight (so a rehearsal refuses for the same
        reasons a real call would) and returns the plan with empty ids.

        Raises:
            KeyError: no such definition in this conversation or the account.
            ValueError: any preflight refusal; nothing was written.
            PromotionIncomplete: a write failed AND its rollback did not finish.
        """

        requested = str(name or "").strip()
        target = str(as_name or requested).strip()
        if not authorized_by:
            # Every promotion is attributable: the index row names the surface the
            # user acted on, and an account write with no origin is exactly what
            # the consent rule (and a later audit) cannot answer for.
            raise ValueError(
                "a promotion must name the surface that authorized it "
                "(surface:desktop, surface:tui, or a host consent origin)"
            )
        with _write_section(self):
            plan = self._promote_plan(requested, target, replace)
            source, prompt, existing = plan.source, plan.prompt, plan.existing
            # The index goes LAST, but its writability is a PREFLIGHT: discovering
            # an unwritable config root after creating the row would roll back
            # work that was otherwise fine.
            role_provenance.require_writable(self.config_dir)
            if dry_run:
                planned = {"kind": "agent", "name": target, "id": ""}
                return PromoteResult(
                    created=[] if existing is not None else [planned],
                    expanded=[],
                    replaced=[target] if existing is not None else [],
                    provenance_ids=[],
                    dry_run=True,
                )
            created: list[dict[str, Any]] = []
            replaced: list[str] = []
            undo: list[tuple[dict[str, Any], Callable[[], None]]] = []
            try:
                if existing is not None:
                    row = {"kind": "agent", "name": target, "id": existing.id}
                    # Registered AFTER the overwrite returns: a failed overwrite
                    # has already restored itself and raised, so there would be
                    # nothing here for an outer undo to do.
                    undo.append((row, self._overwrite_agent(existing, source, target, prompt)))
                    replaced.append(target)
                    record_id = str(existing.id)
                else:
                    row_id, step = self._create_agent(source, target, prompt)
                    row = {"kind": "agent", "name": target, "id": row_id}
                    undo.append((row, step))
                    created.append(row)
                    record_id = row_id
                provenance_ids = role_provenance.append_promotions(
                    self.config_dir,
                    [
                        {
                            "kind": "agent",
                            "id": record_id,
                            "name": target,
                            "source_session": _source_session_id(self),
                            "at": _now_iso(),
                            "authorized_by": authorized_by,
                            "replaced": target if existing is not None else None,
                        }
                    ],
                )
            except BaseException:
                remaining = _undo_all(undo)
                if remaining:
                    raise PromotionIncomplete(
                        _incomplete_message("agent", target, remaining), remaining
                    ) from None
                raise
            return PromoteResult(
                created=created,
                expanded=[],
                replaced=replaced,
                provenance_ids=provenance_ids,
            )

    def _promote_plan(self, requested: str, target: str, replace: bool) -> "_AgentPromotionPlan":
        """Every refusal a promotion can answer with, BEFORE its first write.

        Runs under the caller's write section: the collision check and the write
        must share one lock, or two promotions of the same name both pass it.
        """

        from local_operator.agent_profiles import is_role, is_specialist

        source = self.session_registry.get_agent_by_name(requested)
        if source is None:
            if self.global_registry.get_agent_by_name(requested) is not None:
                raise ValueError(promote_already_available("agent", requested))
            raise KeyError(requested)
        if not (is_role(source) or is_specialist(source)):
            raise ValueError(
                f"an agent named {requested!r} exists in this conversation and is not a "
                "role or specialist, so there is nothing to promote."
            )
        if not target or any(char.isspace() or char in "/\\" for char in target):
            raise ValueError("Profile names cannot contain spaces or path separators")
        if target.casefold() != requested.casefold():
            other = self.session_registry.get_agent_by_name(target)
            if other is not None and other.id != source.id:
                # An account row under a name THIS conversation also holds is the
                # reverse of the S2 shadow rule and just as confusing: the new
                # copy exists everywhere except here.
                raise ValueError(shadow_refusal("agent", target, existing_scope="session"))
        existing = _account_agent_key(self.global_registry, target)
        if existing is not None and not replace:
            raise ValueError(promote_collision("agent", target))
        # Read once, before the first write: a prompt that cannot be read must
        # refuse the promotion rather than produce a copy that runs without it.
        prompt = self.session_registry.get_agent_system_prompt(source.id)
        return _AgentPromotionPlan(source=source, target=target, prompt=prompt, existing=existing)

    def _create_agent(
        self, source: Any, target: str, prompt: str
    ) -> tuple[str, Callable[[], None]]:
        """Write the account copy; a failure after the row landed removes it."""

        saved = self.global_registry.save_agent(_promoted_agent_row(source, target))
        try:
            self.global_registry.set_agent_system_prompt(saved.id, prompt)
        except BaseException as error:
            try:
                self.global_registry.delete_agent(saved.id)
            except Exception as rollback_error:  # noqa: BLE001 — reported below
                raise PromotionIncomplete(
                    f"copying {target!r} into your account wrote the row but not its "
                    f"instructions, and removing the row also failed ({rollback_error}); "
                    f"agent {saved.id} remains in your account.",
                    [
                        {
                            "kind": "agent",
                            "name": target,
                            "id": saved.id,
                            "error": str(rollback_error),
                        }
                    ],
                ) from error
            raise
        return saved.id, lambda: self.global_registry.delete_agent(saved.id)

    def _overwrite_agent(
        self, existing: Any, source: Any, target: str, prompt: str
    ) -> Callable[[], None]:
        """Replace the named account row in place and return its restore step.

        In place, so the row keeps its id — anything already pointing at it (a
        schedule, a roster, a client's open editor) keeps pointing at it. The
        snapshot is taken BEFORE the first write: after a failed prompt write the
        only readable copy of the user's old text is the one taken here.
        """

        old_prompt = self.global_registry.get_agent_system_prompt(existing.id)
        snapshot = _agent_edit_fields(existing)

        def restore() -> None:
            self.global_registry.update_agent(existing.id, snapshot)
            self.global_registry.set_agent_system_prompt(existing.id, old_prompt)

        try:
            self.global_registry.update_agent(existing.id, _agent_edit_fields(source, name=target))
            self.global_registry.set_agent_system_prompt(existing.id, prompt)
        except BaseException as error:
            try:
                restore()
            except Exception as rollback_error:  # noqa: BLE001 — reported below
                raise PromotionIncomplete(
                    f"overwriting agent {target!r} failed and restoring the previous "
                    f"definition also failed ({rollback_error}); agent {existing.id} may "
                    "now hold a mixture of both.",
                    [
                        {
                            "kind": "agent",
                            "name": target,
                            "id": str(existing.id),
                            "error": str(rollback_error),
                        }
                    ],
                ) from error
            raise
        return restore

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

    # -- promotion: copy THIS conversation's team, closure included (S4) ------
    def promote(
        self,
        name: str,
        *,
        as_name: str | None = None,
        replace: bool = False,
        authorized_by: str | None = None,
        agents: Any = None,
        dry_run: bool = False,
    ) -> PromoteResult:
        """Copy the conversation team ``name`` into the account store.

        A team is not usable from another conversation unless its session-scoped
        MEMBERS are there too: a roster naming a role that exists only in the
        origin conversation resolves here and breaks there. So the roster is
        walked through the LAUNCH resolver and every conversation-scoped member —
        and nested conversation team — is copied in the same call, bounded by
        ``teams.MAX_ORG_DEPTH`` and :data:`MAX_PROMOTION_ROWS`. Account rows and
        packaged starters are REUSED: they already resolve everywhere.

        ``agents`` is the agent composite that overlays the SAME conversation.
        It is required: without it the roster cannot be resolved, and resolving
        it against the account store alone would silently pass a team that breaks
        elsewhere.

        All-or-nothing across the whole set: every write is preflighted first,
        and a failure deletes exactly the rows this call created, in reverse.
        Raises exactly what :meth:`ScopedAgentRegistry.promote` documents.
        """

        if agents is None:
            raise ValueError(
                "promoting a team needs this conversation's agent registry: its roster "
                "members must be resolved, and conversation-scoped ones copied, in the "
                "same call"
            )
        if not authorized_by:
            raise ValueError(
                "a promotion must name the surface that authorized it "
                "(surface:desktop, surface:tui, or a host consent origin)"
            )
        from local_operator.teams import validate_team_name

        requested = str(name or "").strip()
        with _write_section(self):
            source = self.session_registry.get_team_by_name(requested)
            if source is None:
                if self.global_registry.get_team_by_name(requested) is not None:
                    raise ValueError(promote_already_available("team", requested))
                raise KeyError(requested)
            target = validate_team_name(str(as_name or requested).strip())
            if target.casefold() != requested.casefold():
                other = self.session_registry.get_team_by_name(target)
                if other is not None and other.id != source.id:
                    raise ValueError(shadow_refusal("team", target, existing_scope="session"))
            existing = self.global_registry.get_team_by_name(target)
            if existing is not None and str(existing.name).casefold() != target.casefold():
                # The target resolves only through another team's ALIAS. A replace
                # would RENAME that row out from under the key its own users know,
                # which is not what the confirmation promised.
                raise ValueError(promote_collision("team", target))
            conflict = _account_team_keys(
                self.global_registry,
                target,
                list(source.aliases or []),
                skip_id=str(existing.id) if existing is not None else None,
            )
            if conflict is not None:
                raise ValueError(promote_collision("team", target))
            if existing is not None and not replace:
                raise ValueError(promote_collision("team", target))

            member_rows, nested_teams = _team_promotion_plan(agents, self, source)
            # Preflight the whole closure BEFORE the first write. The agent
            # composite's own planner answers the collision/shadow/readability
            # questions, so a member is judged by exactly the rule a direct
            # promote of it would be — one rule, one sentence.
            agent_plans = [agents._promote_plan(row.name, row.name, replace) for row in member_rows]
            team_plans: list[tuple[Any, str, Any | None]] = []
            for nested in nested_teams:
                member_target = validate_team_name(str(nested.name))
                member_existing = self.global_registry.get_team_by_name(member_target)
                if member_existing is not None and not replace:
                    raise ValueError(promote_collision("team", member_target))
                member_conflict = _account_team_keys(
                    self.global_registry,
                    member_target,
                    list(nested.aliases or []),
                    skip_id=str(member_existing.id) if member_existing is not None else None,
                )
                if member_conflict is not None:
                    raise ValueError(promote_collision("team", member_target))
                team_plans.append((nested, member_target, member_existing))
            role_provenance.require_writable(self.config_dir)

            planned: list[dict[str, Any]] = [
                {"kind": "agent", "name": plan.target, "id": ""} for plan in agent_plans
            ] + [{"kind": "team", "name": plan[1], "id": ""} for plan in team_plans]
            if existing is None:
                planned.append({"kind": "team", "name": target, "id": ""})
            if dry_run:
                return PromoteResult(
                    created=planned,
                    expanded=[
                        row["name"]
                        for row in planned
                        if not (row["kind"] == "team" and row["name"] == target)
                    ],
                    replaced=[
                        *[plan.target for plan in agent_plans if plan.existing is not None],
                        *[plan[1] for plan in team_plans if plan[2] is not None],
                        *([target] if existing is not None else []),
                    ],
                    provenance_ids=[],
                    dry_run=True,
                )

            created: list[dict[str, Any]] = []
            replaced: list[str] = []
            records: list[dict[str, Any]] = []
            undo: list[tuple[dict[str, Any], Callable[[], None]]] = []
            try:
                # Members first: a team row must never be published pointing at a
                # role another conversation cannot resolve yet.
                for plan in agent_plans:
                    member_target = plan.target
                    if plan.existing is not None:
                        row = {"kind": "agent", "name": member_target, "id": plan.existing.id}
                        undo.append(
                            (
                                row,
                                agents._overwrite_agent(
                                    plan.existing, plan.source, member_target, plan.prompt
                                ),
                            )
                        )
                        replaced.append(member_target)
                        row_id = str(plan.existing.id)
                        replaced_name: str | None = member_target
                    else:
                        row_id, step = agents._create_agent(plan.source, member_target, plan.prompt)
                        row = {"kind": "agent", "name": member_target, "id": row_id}
                        undo.append((row, step))
                        created.append(row)
                        replaced_name = None
                    records.append(
                        {
                            "kind": "agent",
                            "id": row_id,
                            "name": member_target,
                            "source_session": _source_session_id(self),
                            "at": _now_iso(),
                            "authorized_by": authorized_by,
                            "replaced": replaced_name,
                        }
                    )
                for nested, member_target, member_existing in team_plans:
                    if member_existing is not None:
                        row = {"kind": "team", "name": member_target, "id": member_existing.id}
                        undo.append(
                            (row, self._overwrite_team(member_existing, nested, member_target))
                        )
                        replaced.append(member_target)
                        row_id = str(member_existing.id)
                        replaced_name = member_target
                    else:
                        saved, step = self._create_team(nested, member_target)
                        row = {"kind": "team", "name": member_target, "id": saved.id}
                        undo.append((row, step))
                        created.append(row)
                        row_id = saved.id
                        replaced_name = None
                    records.append(
                        {
                            "kind": "team",
                            "id": row_id,
                            "name": member_target,
                            "source_session": _source_session_id(self),
                            "at": _now_iso(),
                            "authorized_by": authorized_by,
                            "replaced": replaced_name,
                        }
                    )
                if existing is not None:
                    row = {"kind": "team", "name": target, "id": existing.id}
                    undo.append((row, self._overwrite_team(existing, source, target)))
                    replaced.append(target)
                    root_id = str(existing.id)
                    root_replaced: str | None = target
                else:
                    saved, step = self._create_team(source, target)
                    row = {"kind": "team", "name": target, "id": saved.id}
                    undo.append((row, step))
                    created.append(row)
                    root_id = saved.id
                    root_replaced = None
                records.append(
                    {
                        "kind": "team",
                        "id": root_id,
                        "name": target,
                        "source_session": _source_session_id(self),
                        "at": _now_iso(),
                        "authorized_by": authorized_by,
                        "replaced": root_replaced,
                    }
                )
                provenance_ids = role_provenance.append_promotions(self.config_dir, records)
            except BaseException:
                remaining = _undo_all(undo)
                if remaining:
                    raise PromotionIncomplete(
                        _incomplete_message("team", target, remaining), remaining
                    ) from None
                raise
            return PromoteResult(
                created=created,
                expanded=[
                    row["name"]
                    for row in created
                    if not (row["kind"] == "team" and row["name"] == target)
                ],
                replaced=replaced,
                provenance_ids=provenance_ids,
            )

    def _create_team(self, source: Any, target: str) -> tuple[Any, Callable[[], None]]:
        """Write the account copy; a failure after the row landed removes it."""

        saved = self.global_registry.create_team(_team_edit_fields(source, name=target))
        return saved, lambda: self.global_registry.delete_team(saved.id)

    def _overwrite_team(self, existing: Any, source: Any, target: str) -> Callable[[], None]:
        """Replace the named account team in place and return its restore step.

        The snapshot carries BOTH briefs (``get_team_by_name`` hydrates them), and
        it is taken before the first write: ``update_team`` rewrites the fields and
        both brief files in one publication, so a failure inside it can only be
        undone from a copy taken here.
        """

        snapshot = _team_edit_fields(existing, name=existing.name)

        def restore() -> None:
            self.global_registry.update_team(existing.id, snapshot)

        try:
            self.global_registry.update_team(existing.id, _team_edit_fields(source, name=target))
        except BaseException as error:
            try:
                restore()
            except Exception as rollback_error:  # noqa: BLE001 — reported below
                raise PromotionIncomplete(
                    f"overwriting team {target!r} failed and restoring the previous "
                    f"definition also failed ({rollback_error}); team {existing.id} may "
                    "now hold a mixture of both.",
                    [
                        {
                            "kind": "team",
                            "name": target,
                            "id": str(existing.id),
                            "error": str(rollback_error),
                        }
                    ],
                ) from error
            raise
        return restore

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


# ---------------------------------------------------------------------------
# Promotion: copy a conversation definition into the account (LOH-9 S4)
# ---------------------------------------------------------------------------

#: The most definitions ONE promotion may create. A team expands into the
#: session-scoped roles and nested session teams its roster needs, so its
#: closure is bounded here as well as by ``teams.MAX_ORG_DEPTH``: depth alone
#: does not bound a WIDE roster of nested teams, each with its own roster.
#: A team row caps its own slots, so 50 is far past any real team and far below
#: anything that would hold the write lock across an unbounded walk.
MAX_PROMOTION_ROWS = 50


@dataclass(frozen=True)
class PromoteResult:
    """What one promotion did — or, with ``dry_run``, would do.

    ``created`` names every row this call wrote, kind by kind. ``expanded`` is
    the subset a TEAM needed (its conversation-scoped members and nested teams),
    which is what a receipt lists so the user knows the copy is not just one row.
    ``replaced`` names the account rows an authorised ``replace`` overwrote —
    exactly what the confirmation promised to overwrite.
    ``provenance_ids`` are the definition ids recorded in the device index, one
    per created or replaced row, so a receipt can be joined to the index.

    A dry run fills the same names with EMPTY ids and no provenance ids: it is
    the rehearsal a surface shows before it asks for confirmation, and it must
    be impossible to mistake for a write.
    """

    created: list[dict[str, Any]]
    expanded: list[str]
    replaced: list[str]
    provenance_ids: list[str]
    dry_run: bool = False


class PromotionIncomplete(ValueError):
    """Some of a promotion's rows could not be rolled back after a failure.

    Raised ONLY when the undo itself failed, and then INSTEAD of the original
    error: the caller must report what is left in the account, because a clean
    failure that left rows behind is exactly the silent partial promotion this
    transaction exists to prevent.
    """

    def __init__(self, message: str, remaining: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.remaining = remaining


@dataclass(frozen=True)
class _AgentPromotionPlan:
    """One role's preflighted copy: what to read, where it lands, what it replaces."""

    source: Any
    target: str
    prompt: str
    existing: Any | None


def promote_missing(kind: str, name: str) -> str:
    """The refusal for a name the conversation does not hold.

    Mirrors the tool's own not-found sentence (``agent_tool._op_show``), because
    a user who mistypes a promote sees the same two-store explanation an ``agent
    show`` would give — and promotion's extra fact, that only the conversation
    that holds a definition can copy it, is what makes the message actionable.
    """

    noun = "role or specialist" if kind == "agent" else "team"
    return (
        f"no {noun} named {name!r} in this conversation or your account. Only the "
        "conversation that holds a definition can promote it, and a definition "
        "created in another conversation is not visible here; a fork of this "
        "conversation starts with a copy of its definitions."
    )


def promote_already_available(kind: str, name: str) -> str:
    """The refusal for a name that resolves only in the account store."""

    return (
        f"{name!r} is already in your account, so every conversation can use it. "
        "There is nothing to promote."
    )


def promote_collision(kind: str, name: str) -> str:
    """The refusal for a target name the account already holds.

    Names the two ways forward rather than only the conflict: overwriting a
    user-owned definition is a decision the user makes, and copying under
    another name is the non-destructive answer when they did not mean THAT row.
    """

    verb = "/agent promote" if kind == "agent" else "/team promote"
    article = "an" if kind == "agent" else "a"
    return (
        f"your account already has {article} {kind} named {name!r}, so promoting would "
        "overwrite it; nothing was copied. Confirm the overwrite (the desktop "
        f"action's replace, or `{verb} {name} replace` in the TUI) and retry, or "
        "promote under another name."
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_session_id(owner: Any) -> str:
    """The session id whose conversation store a composite overlays.

    ``scope_dir`` is ``<session dir>/roles`` and a session directory is named by
    its id (``session_factory`` builds from ``transcript_dir.name``), so the
    parent's name IS the id — no second field to keep in step.
    """

    return Path(owner.scope_dir).parent.name


def _promoted_agent_row(source: Any, target: str) -> Any:
    """The account row for ``source`` under ``target`` — verbatim, new identity.

    A TRANSPORTED copy rather than a new ``AgentEditFields``: the row's tags are
    the role's kind and policy (``role:``, ``class:``, ``seed:``, ``hub:``,
    ``tools:``, ``effort:``, ``delegate:``), so re-deriving them from a tool
    payload — what ``write_profile`` does for an authored create — would drop
    exactly the policy that makes the copy behave like the original.

    Per-copy identity and CHAT state are the only fields reset: ``id``,
    ``created_date``, and the last-message pair. The conversation that authored
    the source keeps its own row and history; the account copy starts with its
    own empty ones.
    """

    now = datetime.now(timezone.utc)
    return source.model_copy(
        update={
            "id": str(uuid.uuid4()),
            "name": target,
            "created_date": now,
            "last_message": "",
            "last_message_datetime": now,
        }
    )


def _agent_edit_fields(row: Any, *, name: str | None = None) -> Any:
    """Every DEFINITION field of ``row`` as edit fields, for an overwrite.

    ``replace`` has to overwrite a user-owned row in place (its id, and anything
    pointing at it, must survive), so it needs the fields rather than a copied
    ``AgentData``. Nothing is left None: this is a whole-definition replacement,
    and a None would mean "leave the old value", which is the one outcome the
    confirmation did not promise.
    """

    from local_operator.agents import AgentEditFields

    return AgentEditFields(
        name=name or row.name,
        label=row.label or "",
        security_prompt=row.security_prompt,
        hosting=row.hosting,
        model=row.model,
        description=row.description,
        tags=list(row.tags or []),
        categories=list(row.categories or []),
        temperature=row.temperature,
        top_p=row.top_p,
        top_k=row.top_k,
        max_tokens=row.max_tokens,
        stop=list(row.stop) if row.stop is not None else None,
        frequency_penalty=row.frequency_penalty,
        presence_penalty=row.presence_penalty,
        seed=row.seed,
        current_working_directory=row.current_working_directory,
    )


def _team_edit_fields(row: Any, *, name: str) -> Any:
    """Every definition field of ``row`` as edit fields, briefs included.

    ``create_team`` is the only writer that treats supplied briefs as
    authoritative for an id this registry has not loaded (``save_team``'s rule is
    the opposite, deliberately), so a CREATE must go through it — passing a
    transported ``Team`` would publish a row whose briefs are empty.
    """

    from local_operator.teams import TeamEditFields

    return TeamEditFields(
        name=name,
        description=row.description,
        manager=row.manager,
        members=list(row.members),
        label=row.label,
        aliases=list(row.aliases or []),
        instructions=row.instructions,
        project=row.project,
        model_suggestion=row.model_suggestion,
    )


def _account_agent_key(registry: Any, name: str) -> Any | None:
    """The account agent whose name key equals ``name``, if any.

    Casefolded, because that is how the resolver's shadow rule compares names: a
    ``Coder`` in the account and a ``coder`` promoted from a conversation would
    be two rows a picker cannot tell apart.
    """

    wanted = str(name or "").strip().casefold()
    for row in registry.list_agents():
        if str(getattr(row, "name", "") or "").strip().casefold() == wanted:
            return row
    return None


def _account_team_keys(
    registry: Any, name: str, aliases: list[str], *, skip_id: str | None = None
) -> Any | None:
    """The account team any of the candidate KEYS would collide with.

    Team lookup resolves by casefolded name with an alias fallback, so a copy
    carrying an alias that matches another row's name is a real collision even
    though its own name is free. ``skip_id`` exempts the row a confirmed
    ``replace`` is about to overwrite: its former keys are replaced by this very
    write, so it cannot collide with itself.
    """

    for key in [name, *aliases]:
        found = registry.get_team_by_name(key)
        if found is not None and str(found.id) != str(skip_id or ""):
            return found
    return None


def _undo_all(undo: list[tuple[dict[str, Any], Callable[[], None]]]) -> list[dict[str, Any]]:
    """Run the rollback steps in REVERSE, returning the ones that failed.

    A failing step must not stop the rest: one stuck row is not a reason to
    leave every later row behind. Each entry pairs the ROW it undoes with its
    step, so the returned remainder names what is still in the account rather
    than an index the caller cannot interpret.
    """

    remaining: list[dict[str, Any]] = []
    for row, step in reversed(undo):
        try:
            step()
        except Exception as error:  # noqa: BLE001 — reported, never swallowed silently
            remaining.append({**row, "error": str(error)})
    return remaining


def _incomplete_message(kind: str, name: str, remaining: list[dict[str, Any]]) -> str:
    return (
        f"promoting {kind} {name!r} failed AND its rollback did not finish, so "
        f"{len(remaining)} written row(s) may still be in your account. Review the "
        "account store before retrying; nothing here reported success."
    )


def _team_promotion_plan(agents: Any, teams: Any, source: Any) -> tuple[list[Any], list[Any]]:
    """The conversation rows a team's copy needs, resolved through the LAUNCH resolver.

    Returns ``(agent_rows, nested_team_rows)``. The walk is exactly the one a
    launch performs (``desktop_profiles.validate_target``): ``profile_detail``
    per manager/member with the same ``MAX_ORG_DEPTH`` bound and the same cycle
    refusal. A preflight that used store membership instead would refuse a
    roster naming a PACKAGED STARTER that simply is not installed yet — a seed is
    resolvable everywhere, so it needs no copy and must not block the promotion.

    A member counts as needing promotion exactly when the resolver says its row
    lives in THIS conversation (``scope == "session"``); everything else — an
    account row, a packaged seed — already resolves in another conversation and
    is reused.
    """

    from local_operator.server.utils.desktop_profiles import profile_detail
    from local_operator.teams import MAX_ORG_DEPTH

    agent_rows: list[Any] = []
    nested_teams: list[Any] = []
    seen_agents: set[str] = set()
    seen_teams: set[str] = set()

    def visit(team: Any, ancestors: tuple[str, ...]) -> None:
        slots = [team.manager, *(m.role for m in team.members if m.kind != "team")]
        for slot in slots:
            try:
                detail = profile_detail(agents, slot, detail=False)
            except KeyError:
                raise ValueError(
                    f"the roster of {team.name!r} names {slot!r}, which does not resolve "
                    "to a role, specialist or packaged starter; the copy would be broken "
                    "in another conversation, so nothing was promoted."
                ) from None
            if detail.get("scope") != "session":
                continue
            row = agents.session_registry.get_agent_by_name(str(detail["name"]))
            if row is None:  # pragma: no cover — the resolver just put it in this store
                continue
            if row.id not in seen_agents:
                seen_agents.add(row.id)
                agent_rows.append(row)
        for member in team.members:
            if member.kind != "team":
                continue
            if member.role in ancestors or len(ancestors) >= MAX_ORG_DEPTH:
                raise ValueError(
                    "the team roster contains a cycle or exceeds the nesting limit "
                    f"({MAX_ORG_DEPTH}), so nothing was promoted."
                )
            nested = teams.get_team_by_name(member.role)
            if nested is None:
                raise ValueError(
                    f"the roster of {team.name!r} nests a team named {member.role!r} which "
                    "does not resolve in this conversation or your account, so nothing "
                    "was promoted."
                )
            if teams.scope_of(nested) != "session":
                continue
            visit(nested, (*ancestors, team.name))
            if nested.id not in seen_teams:
                seen_teams.add(nested.id)
                nested_teams.append(nested)

    visit(source, ())
    if len(agent_rows) + len(nested_teams) + 1 > MAX_PROMOTION_ROWS:
        raise ValueError(
            f"this team expands into {len(agent_rows) + len(nested_teams) + 1} definitions, "
            f"past the {MAX_PROMOTION_ROWS}-definition bound for one promotion; promote "
            "the members it needs first, or copy the team on the desktop after "
            "narrowing its roster."
        )
    return agent_rows, nested_teams


__all__ = [
    "MAX_PROMOTION_ROWS",
    "PromoteResult",
    "PromotionIncomplete",
    "SESSION_ROLES_DIRNAME",
    "ScopedAgentRegistry",
    "ScopedTeamRegistry",
    "promote_missing",
    "scope_agent_registry",
    "scope_dir_for",
    "scope_session_registries",
    "scope_team_registry",
]

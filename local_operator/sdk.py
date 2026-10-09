"""``local_operator.sdk`` — the typed embedding surface over the session machinery.

WHAT THIS IS. One facade over the machinery the five existing consumers already
drive — ``session_factory.create_session`` (construction), the session runtime's
``engage_runtime`` / ``spawn_owned_session`` (delivery, detached or in-process)
and ``SessionProtocol`` (driving and observation). ``lop exec``, the TUI, the
desktop server, the mobile relay and the benchmark apparatus all sit on that
same machinery; the SDK exists so a *programmatic* caller stops reaching for the
primitives — an ``argparse.Namespace`` here, a hand-built tool list there — and
joins the same path instead. The value is precisely that nothing downstream is
new: caching, compaction, failover, delegation, skills, guides, MCP and
approvals are the session's behaviour, and they arrive through the same objects
they arrive through everywhere else.

THE RULE FOR EDITORS: **``sdk.py`` may adapt shape, never semantics.** Any
behavioural change belongs in ``Session`` / ``harness`` / ``session_factory``,
where every consumer sees it — a change made here would make the SDK a second,
divergent implementation, which is the failure mode the design (and the task
that produced it) exists to prevent. Do not add a loop, a tool dispatcher, a
subagent runner, a provider client, or a place where an ``AgentEvent`` is
re-shaped. The event stream's JSON projection for line-oriented consumers stays
``headless_print.printable_event``; the SDK does not invent a second
serialization.

ISOLATION IS A HARD DEFAULT, NOT A FLAG. Session construction goes through
explicit :class:`~local_operator.session.spec.SessionRoots`; the facade scopes
``LOCAL_OPERATOR_CONFIG_DIR`` / ``LOCAL_OPERATOR_HOME`` to those roots around
construction (and around spawns, whose children inherit the scoped environment
instead of the caller's), asserts the roots the environment actually resolves,
refuses a second, different root while another is live in the process, and
refuses the operator's own uid-default roots (or a store under ``/tmp`` /
``$TMPDIR``) unless the caller opted in explicitly. The cache root is checked
by its real resolver too: it derives from ``$HOME`` independently of the two
overrides, so a run whose ``HOME`` is not redirected is told, loudly, rather
than left looking isolated (the incident class from the design's §4).

IMPORT CONTRACT. This module imports the standard library plus
``local_operator.paths`` and ``local_operator.session.spec`` at module scope.
Every engine import is **function-local** — the same discipline
``exec_mode._make_default_session_factory`` and
``serving.spawn_owned_session`` follow, and for the same reason: touching this
module must not put the composition root on anybody's import graph. The
re-exports below are served by PEP 562 ``__getattr__`` (the pattern
``local_operator/providers/__init__.py`` uses), and ``tests/unit/test_sdk.py``
pins both halves.

WHAT PR 1 IS, AND WHAT IT DEFERS — stated here because the surface is
published additive-stable and the gaps must not be discovered by reading the
code:

* ``open_session`` (own mode) is the full in-process surface: spec → the exec
  helpers (``exec_startup.resolve_startup`` / ``declared_tool_inventory``) →
  ``create_session`` → the post-open attachment sequence → the approval policy.
* ``open_session(mode="attach")`` opens the viewer path for a session whose
  runtime is ALREADY live (``create_session(has_ui=True)``'s
  ``AttachedSession``). A cold id is refused with a remedy; auto-spawning a
  runtime to attach to is deferred.
* ``spawn_session`` / ``deliver`` drive ``engage_runtime`` against the explicit
  root and carry what the spawn contract can carry (``resume``, ``hosting``,
  ``model``, ``birth_effort``, ``notifications``). The post-open selectors
  (``team`` / ``profile`` / ``tools`` / ``name`` / ``goal`` / non-default
  ``approvals`` / ``yolo``) are refused loudly, not ignored: the runtime child
  is composed from its environment, and there is no sanctioned channel for
  them yet. A caller who needs attachments on a detached session composes:
  open in-process, attach, dispose, then ``spawn_session`` on the same id —
  resume restores the attachment sidecars. The single-call sugar awaits the
  benchmark pilot (PR 2), whose needs should pin its shape.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

from local_operator.paths import AGENT_HOME_ENV, CONFIG_DIR_ENV
from local_operator.session.spec import (
    ApprovalPolicy,
    SessionIsolationError,
    SessionRoots,
    SessionSpec,
    SessionSpecError,
)

if TYPE_CHECKING:  # the names `__getattr__` serves, for type checkers only
    from local_operator.harness.types import AgentEvent, EventHandler
    from local_operator.output_contract import MarkdownSchema, decode_output
    from local_operator.session.protocol import SessionProtocol, ViewerSessionProtocol
    from local_operator.session.runtime.launch import (
        EngageOutcome,
        Errand,
        PeerMessageErrand,
        PromptErrand,
        SteerErrand,
        WakeErrand,
        WarmErrand,
    )

__all__ = [
    "AgentEvent",
    "ApprovalPolicy",
    "EngageOutcome",
    "Errand",
    "EventHandler",
    "MarkdownSchema",
    "PeerMessageErrand",
    "PromptErrand",
    "SessionEventStream",
    "SessionIsolationError",
    "SessionOpenRefused",
    "SessionRoots",
    "SessionSpec",
    "SessionSpecError",
    "SteerErrand",
    "ViewerSessionProtocol",
    "WakeErrand",
    "WarmErrand",
    "decode_output",
    "deliver",
    "events",
    "open_session",
    "spawn_session",
    "SessionProtocol",
]

#: Where each deferred re-export lives. Resolved on first access and cached in
#: the module dict, exactly like ``local_operator/providers/__init__.py`` — the
#: engagement vocabulary pulls the runtime package, the event types pull
#: pydantic, and the protocols pull the session API; none of that may be the
#: price of ``import local_operator.sdk``. ``tests/unit/test_sdk.py`` asserts
#: every name here resolves and that ``import local_operator.sdk`` leaves the
#: heavy set absent.
_LAZY_REEXPORTS: dict[str, str] = {
    "PromptErrand": "local_operator.session.runtime.launch",
    "SteerErrand": "local_operator.session.runtime.launch",
    "PeerMessageErrand": "local_operator.session.runtime.launch",
    "WakeErrand": "local_operator.session.runtime.launch",
    "WarmErrand": "local_operator.session.runtime.launch",
    "Errand": "local_operator.session.runtime.launch",
    "EngageOutcome": "local_operator.session.runtime.launch",
    "AgentEvent": "local_operator.harness.types",
    "EventHandler": "local_operator.harness.types",
    "MarkdownSchema": "local_operator.output_contract",
    "decode_output": "local_operator.output_contract",
    "SessionProtocol": "local_operator.session.protocol",
    "ViewerSessionProtocol": "local_operator.session.protocol",
}


def __getattr__(name: str) -> Any:
    """Resolve a deferred re-export on first access, then cache it.

    Only names in :data:`_LAZY_REEXPORTS` resolve here; anything else is an
    ``AttributeError`` naming the module, so a typo fails like a typo instead
    of returning a falsy object a caller would pass onward.
    """
    module_path = _LAZY_REEXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


class SessionOpenRefused(RuntimeError):
    """A session may not be opened from where this call is running.

    Today the one source is the agent-shell rule (``agent_shell.py``): a
    command an agent's tool call started may not open top-level sessions unless
    its session holds ``task`` (the 2026-09-19 relaxation) — and without this
    guard the SDK would be a way around a rule the CLI enforces. The message is
    the CLI's own, verbatim, so the two cannot describe one rule two ways.
    """


#: The notification kill switch, read FRESH by ``tui.notify`` on every send.
#: Spelled here (and drift-checked by a test against
#: ``tui.notify.ENV_DISABLE``) rather than imported, because importing
#: ``tui.notify`` pulls the terminal and settings modules, and this module
#: keeps its import graph small.
_NO_NOTIFICATIONS_ENV = "LOCAL_OPERATOR_NO_NOTIFICATIONS"

#: One entry per session-root group currently LIVE in this process (root →
#: refcount). The single-root rule (§4.3 of the design) is enforced against
#: THIS table rather than declared: a process must not host sessions from two
#: different roots simultaneously, because components resolve ``paths.*`` at
#: call time and the environment can only be scoped to one root at a time.
#: ``open_session`` registers for its whole ``async with`` body; ``deliver``
#: and ``spawn_session`` register for the engagement's duration — their scope
#: holds the process environment across awaits too, so they must make a
#: concurrent different-root call refuse exactly as a live session does.
_ACTIVE_ROOTS: dict[Path, int] = {}

#: Stream terminator. A sentinel rather than ``None`` so a session can never
#: end a stream by emitting an event shaped like the shutdown marker.
_STREAM_CLOSED: Any = object()


# ---------------------------------------------------------------------------
# Roots: scoping, assertions, single-root enforcement
# ---------------------------------------------------------------------------


def _guard_agent_shell() -> None:
    """Apply the CLI's agent-shell policy to session-creating calls.

    Reads ``agent_shell.py``'s live inventory rather than re-implementing it:
    the refusal and its allowance (``task`` in the session's own tool inventory,
    or the documented QA escape) are one fact in one place, so a change there
    reaches the SDK without an edit here.
    """
    from local_operator.agent_shell import exec_session_refusal

    refusal = exec_session_refusal()
    if refusal:
        raise SessionOpenRefused(refusal)


def _check_single_root(roots: SessionRoots, *, allow_multi_root: bool) -> None:
    """Refuse a second, different root while another one is live.

    Checked against :data:`_ACTIVE_ROOTS`, which tracks the roots of live
    operations — open sessions and in-flight deliveries alike; a process that
    has finished every one of them may move on to another root. Registration
    goes through :func:`_live_root` (never a bare call to this), so the check
    and the registration stay one synchronous step. ``allow_multi_root=True``
    is the deliberate escape for a caller that knows what it is doing (a test
    harness for two stores, or a migration tool); it is a parameter rather
    than a default because the failure it prevents is silent — two live
    roots' lazy resolvers disagreeing about which store they are reading.
    """
    mine = roots.config_path
    others = sorted(str(path) for path in _ACTIVE_ROOTS if path != mine)
    if others and not allow_multi_root:
        raise SessionIsolationError(
            f"this process already hosts a session on {others[0]!r}; one SessionRoots per "
            f"process is the contract (components resolve paths from the environment at call "
            f"time). Finish that session, or pass allow_multi_root=True if the overlap is "
            f"deliberate."
        )


@contextmanager
def _live_root(roots: SessionRoots, *, allow_multi_root: bool) -> Iterator[None]:
    """Register ``roots`` as live for the duration; refuse a second, different root.

    Every path that scopes the process environment to a root enters through
    here — ``open_session`` for its ``async with`` body, ``deliver`` and
    ``spawn_session`` for the engagement — because the environment can carry
    one root at a time, and a delivery's scope holds it across awaits exactly
    as a live session does. The check and the registration are ONE synchronous
    step (nothing may ``await`` between them), which is what makes two
    different roots live at once unreachable rather than unlikely. Refcounted
    per root: same-root overlap is allowed — the environment layers compose
    (see :func:`_scoped_process_env`) — and the last exit clears the entry.
    """
    _check_single_root(roots, allow_multi_root=allow_multi_root)
    key = roots.config_path
    _ACTIVE_ROOTS[key] = _ACTIVE_ROOTS.get(key, 0) + 1
    try:
        yield
    finally:
        _ACTIVE_ROOTS[key] -= 1
        if _ACTIVE_ROOTS[key] <= 0:
            del _ACTIVE_ROOTS[key]


#: The sentinel a scope layer uses when a key must be ABSENT for its duration
#: (the ``CMUX_*``/``LOP_*`` strips) — not ``None``, a real snapshot value
#: meaning "was absent" on :data:`_ENV_BASE`.
_ENV_REMOVE: Any = object()

#: Per-key layers of every live ``_scoped_process_env`` scope: key → [(scope,
#: desired value or ``_ENV_REMOVE``), ...] in entry order. A scope is NOT a
#: plainly nested ``with`` — same-root operations overlap across ``await``
#: points and may exit out of order (a delivery finishing while a sibling is
#: mid-flight is ordinary), and under a save/restore pair the earlier exit
#: would restore values captured before the sibling entered: the sibling then
#: reads a root it did not declare, and the later exit leaks it past both
#: scopes. Each scope therefore owns a layer and the effective value is the
#: TOPMOST layer's, so exits compose exactly in any order. Different-root
#: overlap never reaches here — :func:`_live_root` refuses it first.
_ENV_LAYERS: dict[str, list[tuple["_EnvScope", Any]]] = {}

#: The value each layered key returns to once its LAST layer is gone (``None``
#: = the key was absent). Captured at the first layer's entry — the only value
#: every exit ordering agrees on.
_ENV_BASE: dict[str, str | None] = {}


def _apply_env_layer(key: str) -> None:
    """Set ``key`` from its topmost live layer, or its base value once none live."""
    layers = _ENV_LAYERS[key]
    desired: Any = layers[-1][1] if layers else _ENV_BASE[key]
    if desired is _ENV_REMOVE or desired is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = desired


class _EnvScope:
    """One environment-scope layer; see :data:`_ENV_LAYERS` for the model."""

    __slots__ = ("desired",)

    def __init__(self) -> None:
        self.desired: dict[str, Any] = {}

    def __enter__(self) -> "_EnvScope":
        for key, desired in self.desired.items():
            if key not in _ENV_LAYERS:
                _ENV_BASE[key] = os.environ.get(key)
                _ENV_LAYERS[key] = []
            _ENV_LAYERS[key].append((self, desired))
            _apply_env_layer(key)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for key in self.desired:
            layers = _ENV_LAYERS[key]
            layers[:] = [entry for entry in layers if entry[0] is not self]
            _apply_env_layer(key)
            if not layers:
                del _ENV_LAYERS[key]
                del _ENV_BASE[key]


@contextmanager
def _scoped_process_env(
    roots: SessionRoots, *, for_child: bool = False, notifications: bool | None = None
):
    """Point the process environment at ``roots`` for the duration, then restore.

    This is the mechanism the design pins for components that still resolve
    ``paths.config_dir()``/``paths.agent_home_dir()`` at call time (skill roots,
    caches, profile seeds): scope, let the real resolvers answer, and assert the
    answers. The scope is the PROCESS environment, so it must be held by one
    root at a time — that is what :func:`_live_root` enforces — and it keeps a
    refcounted per-key layer (see :data:`_ENV_LAYERS`) rather than a
    save/restore pair, because more than one same-root operation may be
    suspended inside its scope at once.

    ``for_child=True`` additionally prepares the environment a SPAWNED child
    will inherit — engage spawns the runtime itself, so this is the only place
    the child's environment can be fixed:

    * ``HOME`` becomes ``roots.agent_home``. Every ``~``-derived path in the
      child (the model-listing cache is the one that bites) then lands inside a
      root the caller declared. Deriving ``HOME`` from ``config_dir.parent`` is
      tempting for home-shaped layouts and wrong for the rest: it can leave the
      declared roots entirely, which is how the real home gets touched.
    * ``CMUX_*`` and ``LOP_*`` are stripped — the two prefixes a lop parent
      exports that are read by the CHILD product rather than only by a terminal
      (AGENTS.md §Isolating a run: a headless TUI that inherits
      ``CMUX_WORKSPACE_ID`` renames the operator's real cmux workspaces, and
      ``LOP_MOBILE_CHILD_*`` silently steers which session a child believes it
      is). Both are re-established deliberately by the spawn path below. The
      strip masks every such key a LIVE sibling layer knows about, not only
      the ones present at this scope's entry, so a sibling exiting first
      cannot un-strip the child mid-flight.
    * ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` is set unless the caller opted in —
      "nobody is watching" is the honest default for a spawned SDK session, and
      the switch is read fresh by every notify leg in the child.
    """
    scope = _EnvScope()
    scope.desired[CONFIG_DIR_ENV] = str(roots.config_path)
    scope.desired[AGENT_HOME_ENV] = str(roots.agent_home_path)
    if for_child:
        scope.desired["HOME"] = str(roots.agent_home_path)
        # The strip's key set: everything matching the two prefixes in the
        # environment now OR in a live layer — the second half is what keeps a
        # key masked after the sibling that first removed it exits.
        for key in {*os.environ, *_ENV_LAYERS}:
            if key.startswith(("CMUX_", "LOP_")):
                scope.desired[key] = _ENV_REMOVE
        if notifications is not True:
            scope.desired[_NO_NOTIFICATIONS_ENV] = "1"
    with scope:
        yield


def _under(path: Path, parent: Path) -> bool:
    """Whether ``resolved`` ``path`` equals or sits under ``parent`` (resolved).

    Compared resolved on BOTH sides: the /tmp -> /private/tmp symlink on macOS
    is exactly the case a spelled-path prefix check misses.
    """
    return path == parent or parent in path.parents


def _check_resolved_roots(roots: SessionRoots) -> None:
    """Assert the environment resolves to the declared roots; fail loudly.

    Three checks, in the order the design states them:

    1. The uid-default roots are refused unless the caller opted in
       (``allow_ambient=True``) — a programmatic session that "just happens"
       to target the operator's real store is the incident class this whole
       module's isolation exists for, and making the caller SAY SO is the
       cheapest way to keep it unrepresentable by accident. "Default" is
       answered against the uid's passwd home (``uid_home_dir``), never
       ``Path.home()``: an isolated run's ``$HOME`` lies, and comparing against
       it would conclude the default and reuse the real paths.
    2. The resolved config dir and agent home equal what was declared. They can
       only differ if some component ignored the scoped environment; when one
       does, this raises naming what it resolved instead of proceeding.
    3. The cache root — which derives from ``$HOME`` independently of the two
       overrides, so the scoping above cannot force it — must land under a
       declared root, under ``agent_home``, or under a ``HOME`` that is itself
       redirected away from the uid default. A cache resolving to the operator's
       real home while the session believes it is isolated is precisely the
       "looks isolated" state the design refuses; the remedy is in the message
       (redirect ``HOME``, the reliable method from AGENTS.md, or opt in).
    """
    from local_operator import paths
    from local_operator.model.catalogue import default_cache_dir

    if not roots.allow_ambient:
        if roots.config_path == roots.default_config_path:
            raise SessionIsolationError(
                f"SessionRoots.config_dir {roots.config_path} is this uid's DEFAULT config "
                "dir — the operator's real store. Pass SessionRoots(..., allow_ambient=True) "
                "to target it deliberately, or point the session somewhere else."
            )
        if roots.agent_home_path == roots.default_agent_home_path:
            raise SessionIsolationError(
                f"SessionRoots.agent_home {roots.agent_home_path} is this uid's DEFAULT "
                "agent home; pass allow_ambient=True to target it deliberately."
            )

    resolved_config = paths.config_dir().expanduser().resolve()
    if resolved_config != roots.config_path:
        raise SessionIsolationError(
            f"configuration resolved to {resolved_config}, not the declared "
            f"{roots.config_path}: something ignored the scoped environment. Refusing to "
            f"continue — this is the misconfiguration class the roots exist to make "
            f"impossible, not a warning to log."
        )
    resolved_agent_home = paths.agent_home_dir().expanduser().resolve()
    if resolved_agent_home != roots.agent_home_path:
        raise SessionIsolationError(
            f"agent home resolved to {resolved_agent_home}, not the declared "
            f"{roots.agent_home_path}: something ignored the scoped environment."
        )

    if roots.allow_ambient:
        return
    cache_root = default_cache_dir().resolve()
    if _under(cache_root, roots.config_path) or _under(cache_root, roots.agent_home_path):
        return
    uid_home = roots.default_config_path.parent  # uid home, resolved
    process_home = os.environ.get("HOME")
    redirected = bool(process_home) and Path(process_home).expanduser().resolve() != uid_home
    if redirected:
        # The caller redirected HOME (the reliable isolation method); the cache
        # then sits under the scratch home, not under the operator's.
        return
    raise SessionIsolationError(
        f"the model-listing cache resolves to {cache_root}, outside the declared roots, "
        f"while HOME is this uid's real home — a run like this reads and writes the "
        f"operator's cache while looking isolated. Redirect HOME to the session's scratch "
        f"home (AGENTS.md §Isolating a run), or pass SessionRoots(..., allow_ambient=True)."
    )


# ---------------------------------------------------------------------------
# Approval policy installation
# ---------------------------------------------------------------------------


def _install_approval_policy(session: Any, policy: ApprovalPolicy) -> None:
    """Install the gates ``policy`` names, via ``set_approval_handler``.

    The mapping table is :class:`ApprovalPolicy`'s docstring; the enforcement
    points are the session's own (``Session._tool_approval_gate`` consults a
    declared-and-unattended inventory before the base gate; ``yolo`` skips the
    gate object entirely), so this function only chooses WHAT to install:

    * ``refuse`` — a two-argument gate that raises
      ``ApprovalUnavailableError``. Typed, like exec's non-tty gate: the call
      sites render "nobody could ask" with its remedies rather than the
      "user denied" copy meant for a person who answered.
    * ``auto`` — an always-true gate, the same posture ``--yolo`` installs.
    * ``callback`` — the caller's function, exactly as a full front end's.
    * ``declared`` — the inventory (installed earlier, ``unattended=True``)
      answers its own members; the base gate installed here is the typed
      refusal, so anything that reaches the gate is refused the honest way.
      Deterministic on purpose: the SDK installs refusal rather than letting a
      developer's tty inherit exec's interactive y/N prompt, because a library
      must not block on a terminal its caller never agreed to lend.
    """
    from local_operator.harness.approval import ApprovalUnavailableError

    if policy.mode == "callback":
        session.set_approval_handler(policy.handler)
        return
    if policy.mode == "auto":
        session.set_approval_handler(_auto_approve_gate)
        return
    session.set_approval_handler(_make_refuse_gate(ApprovalUnavailableError))


async def _auto_approve_gate(tool_name: str, description: str) -> bool:
    """The ``auto()`` posture: every gated call is approved."""
    return True


def _make_refuse_gate(error_type: type[Exception]) -> Callable[..., Any]:
    """Build the ``refuse()`` gate bound to the harness's typed refusal.

    A closure rather than a module-level function so the ``local_operator.harness``
    import stays function-local (see the module's import contract); the
    signature is deliberately the narrow two-argument shape, which
    ``harness.approval._job_id_style`` reads and passes accordingly.
    """

    async def refuse(tool_name: str, description: str) -> bool:
        raise error_type(
            tool_name,
            "this SDK session was opened with ApprovalPolicy.refuse() and no approval "
            "surface is attached",
        )

    return refuse


# ---------------------------------------------------------------------------
# Construction — the same helpers exec runs, in exec's order
# ---------------------------------------------------------------------------


async def _build_session(spec: SessionSpec, roots: SessionRoots, *, mode: str) -> Any:
    """Build one session from ``spec`` — construction plus post-open attachment.

    The sequence mirrors ``exec_session.run_session``'s call into
    ``exec_startup``: ``resolve_startup`` validates and resolves the team
    against THIS root before anything is constructed, then, post-open, team →
    profile → tool inventory → goal → name. It calls the same ``Session``
    methods ``apply_startup`` calls, in the same order, rather than
    ``apply_startup`` itself for one reason, stated so nobody "fixes" it back:
    ``apply_startup`` derives a declared inventory's stand-as-approval from
    ``not control and not stdin.isatty()`` — a *terminal* — and an SDK process
    has no terminal whose state may speak for its caller. The SDK's
    ``ApprovalPolicy`` is the deterministic source for that bit
    (``stands_as_approval``), and installing the inventory through
    ``set_tool_inventory`` is one-way, so the decision has to be made at the
    single call, not corrected afterwards.
    """
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.exec_startup import declared_tool_inventory, resolve_startup
    from local_operator.session_factory import HostingNotConfiguredError, create_session

    runner_args = spec.to_runner_args()
    # Resolves the team/profile names against the SCOPED config dir, so a
    # scratch root cannot resolve the operator's teams by accident.
    try:
        team = resolve_startup(runner_args)
    except ValueError as error:
        # The preflight is SHARED with the CLI, so it refuses in the CLI's
        # exception type — a plain ``ValueError``, e.g. ``--profile`` combined
        # with ``--team`` since issue #2014, or a name that does not resolve.
        # The SDK promises its OWN type for invalid input (``SessionSpecError``,
        # what a caller writes ``except`` for), so it is re-raised here rather
        # than changed at the shared seam, which would move the CLI's contract
        # too (agent review round 1, NIT-2).
        raise SessionSpecError(str(error)) from error

    config_manager = ConfigManager(config_dir=roots.config_path)
    agent_registry = AgentRegistry(config_dir=roots.config_path)

    # ``Any`` on purpose: ``create_session``'s declared return is
    # ``SessionProtocol``, and the attachment calls below (``attach_team``,
    # ``attach_agent_profile``, ``set_tool_inventory``) are concrete-Session
    # members the protocol deliberately omits — ``exec_session.apply_startup``
    # takes the same view for the same reason.
    try:
        session: Any = await create_session(
            spec.to_namespace(),
            config_manager,
            agent_registry,
            has_ui=(mode == "attach"),
            cwd=str(roots.cwd_path),
        )
    except HostingNotConfiguredError as error:
        if mode != "attach":
            raise
        # The factory's hosting preflight runs BEFORE the attach branch below
        # can read ``owns_runtime`` (a cold id falls through to the full local
        # build), so in a root with no hosting configured the remedy the attach
        # contract promises would never run. Refused with both facts: no live
        # runtime is serving the id, and this root cannot start one until its
        # hosting is configured — which is also why spawn_session()/deliver()
        # cannot be the next step here yet.
        raise SessionSpecError(
            f"no live runtime is serving {spec.resume!r}, and this root cannot start "
            f"one: {error} Configure hosting for this root (its config.yml), then "
            "engage the id with spawn_session()/deliver() and attach once a runtime "
            "exists."
        ) from error

    if mode == "attach":
        # A viewer, or nothing. create_session(has_ui=True) hands back an
        # AttachedSession only when a live runtime already owns the id; a cold
        # id falls through to an in-process owner, which is NOT what
        # mode="attach" promises — dispose it and say so rather than
        # silently returning something that will write.
        if getattr(session, "owns_runtime", False):
            await session.dispose()
            raise SessionSpecError(
                f"no live runtime is serving {spec.resume!r}; open it with "
                "mode='own' to run it in this process, or engage it first with "
                "spawn_session()/deliver() and attach once a runtime exists."
            )
        return session

    if team is not None:
        session.attach_team(team)
    if spec.profile:
        if not session.attach_agent_profile(spec.profile):
            raise SessionSpecError(f"Could not attach profile {spec.profile!r}")
    inventory = declared_tool_inventory(session, runner_args)
    if spec.approvals.mode == "declared":
        # The policy's own list is the declaration when spec.tools does not
        # name one — the design's example spells the bound as
        # ``approvals=ApprovalPolicy.declared([...])`` and nothing else. If
        # BOTH name a list they must agree: two spellings of one bound are the
        # second place for it to drift, and the session refuses a widening
        # re-declaration anyway (set_tool_inventory is one-way), so agreeing
        # here is what keeps that refusal from becoming a startup failure.
        policy_tools = spec.approvals.tools
        if inventory is None:
            inventory = policy_tools
        elif tuple(inventory) != policy_tools:
            raise SessionSpecError(
                f"spec.tools {tuple(inventory)} and ApprovalPolicy.declared({policy_tools}) "
                "name different inventories; state the run's reach once"
            )
    if inventory is not None:
        session.set_tool_inventory(inventory, unattended=spec.approvals.stands_as_approval)
    if spec.goal is not None:
        session.set_goal(spec.goal)
    if spec.name is not None:
        session.set_conversation_name(spec.name)
    if spec.output_format is not None:
        # The output contract, applied through the same post-open session
        # method exec installs it with — see the docstring above for why this
        # path calls the method rather than ``apply_startup``. Validation is
        # ``OutputContract``'s (the one validator both paths share); a refused
        # contract surfaces as ``SessionSpecError``, the spec surface's own
        # error type, so callers catch one class.
        from local_operator.output_contract import (
            OutputContract,
            OutputContractError,
            OutputFormat,
        )

        try:
            contract = OutputContract(
                # ``SessionSpec.__post_init__`` has already refused any value
                # outside its ``_OUTPUT_FORMATS`` mirror of this vocabulary, so
                # the cast is the annotation catching up with a CHECKED value —
                # ``OutputContract.__post_init__`` re-checks it regardless.
                format=cast(OutputFormat, spec.output_format),
                schema=spec.output_schema,
                retries=2 if spec.output_retries is None else spec.output_retries,
            )
        except OutputContractError as error:
            raise SessionSpecError(str(error)) from error
        session.set_output_contract(contract)
    _install_approval_policy(session, spec.approvals)
    return session


def _assert_session_dir_under_root(session: Any, roots: SessionRoots) -> None:
    """The store-path canary: the created session's directory must be in-root.

    The transcript path is the one durable artifact the session itself reports,
    so it is what "prove it inherited the root" checks against — not a config
    read. A session whose transcript landed outside the declared root is the
    silent-misconfiguration shape this facade exists to refuse.
    """
    transcript_path = getattr(session, "transcript_path", None)
    if transcript_path is None:  # a reduced host with an in-memory store
        return
    directory = Path(transcript_path).parent.resolve()
    if not (_under(directory, roots.config_path) or _under(directory, roots.agent_home_path)):
        raise SessionIsolationError(
            f"the session's own directory {directory} landed outside the declared roots; "
            f"refusing to hand back a session that is not in the store you asked for."
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


@asynccontextmanager
async def open_session(
    spec: SessionSpec,
    *,
    roots: SessionRoots,
    mode: Literal["own", "attach"] = "own",
    allow_multi_root: bool = False,
) -> AsyncIterator["SessionProtocol"]:
    """Open a session against explicit ``roots``; yield it; dispose on exit.

    ``mode="own"`` builds the session in this process (``has_ui=False``) — the
    same shape as ``lop exec`` foreground: if the caller dies, the turn dies.
    ``mode="attach"`` opens the viewer path for an id whose runtime is already
    live (``spec.resume`` is required; every other spec field is refused) and
    yields the ``AttachedSession``; a cold id is refused with a remedy rather
    than silently becoming an owner.

    The environment is scoped to ``roots`` for the whole ``async with`` body:
    construction, the caller's turns, and every lazy resolver in between see
    the same root, and it is restored on exit. One process may hold one root at
    a time (``allow_multi_root=True`` to override) — an in-flight delivery
    holds its root the same way; see :func:`_live_root`.

    Exit ``dispose()``s the session. For an owner that ends the turn loop and
    releases the claim (as closing exec does); for a viewer it drops the
    viewer's side without touching the owner.
    """
    if mode not in ("own", "attach"):
        raise SessionSpecError(f"mode must be 'own' or 'attach', not {mode!r}")
    if mode == "attach":
        if spec.resume is None:
            raise SessionSpecError("mode='attach' needs spec.resume: a viewer attaches to an id")
        _refuse_attach_extras(spec)
    else:
        _guard_agent_shell()

    roots.assert_durable()
    with _live_root(roots, allow_multi_root=allow_multi_root):
        with _scoped_process_env(roots):
            _check_resolved_roots(roots)
            session = await _build_session(spec, roots, mode=mode)
            try:
                if mode == "own":
                    _assert_session_dir_under_root(session, roots)
                yield session
            finally:
                await session.dispose()


def _refuse_attach_extras(spec: SessionSpec) -> None:
    """Attach opens a viewer; nothing but ``resume`` may be set on the spec.

    A viewer observes and steers a session someone else owns. Attaching a team,
    bounding tools, naming the conversation, or carrying an approval policy are
    OWNER edits: the attach path returns before ``_install_approval_policy``,
    so a non-default policy accepted here would never be installed and the
    caller would be left believing in a gate nothing consults. Refused with the
    remedy named rather than ignored: a caller who believes they set a policy
    and finds it inert is the failure shape this SDK refuses to have. The one
    policy attach accepts is the default ``refuse()`` — the preset of a spec
    that never mentioned approvals — and every other preset is refused.
    """
    set_fields = [
        name
        for name, value in (
            ("hosting", spec.hosting),
            ("model", spec.model),
            ("agent_name", spec.agent_name),
            ("agent_id", spec.agent_id),
            ("team", spec.team),
            ("profile", spec.profile),
            ("tools", spec.tools),
            ("name", spec.name),
            ("goal", spec.goal),
            ("birth_effort", spec.birth_effort),
        )
        if value is not None
    ]
    occupancy = [
        name
        for name, value in (
            ("output_format", spec.output_format),
            ("output_schema", spec.output_schema),
            ("output_retries", spec.output_retries),
        )
        if value is not None
    ]
    if occupancy:
        # Enforced by the OWNER runtime, never the viewer: an attach spec that
        # carried enforcement would be inert by construction (the viewer
        # cannot install it), and an inert setting a caller believes in is the
        # failure shape this surface refuses to have.
        raise SessionSpecError(
            "output enforcement cannot be set on an attached session; the owning " "runtime decides"
        )
    # Value equality, deliberately: ``refuse()`` is the default, so a caller
    # who spells it out is indistinguishable from one who does not — and every
    # other preset (auto/declared/callback) is refused, each enum value covered
    # by the parametrized attach test.
    if spec.approvals != ApprovalPolicy.refuse():
        set_fields.append("approvals")
    if spec.yolo or spec.train or spec.workstream or set_fields:
        raise SessionSpecError(
            "mode='attach' opens a viewer for spec.resume; it does not attach teams, "
            "profiles, tools, names, goals or approval policies — that is the "
            f"owner's side. Set only resume on this spec (or use mode='own'). "
            f"Offending fields: "
            f"{', '.join(set_fields) if set_fields else 'yolo/train/workstream'}."
        )


def _refuse_spawn_extras(spec: SessionSpec) -> None:
    """Refuse spec fields the spawn contract cannot deliver, loudly.

    A spawned runtime child is composed from its environment (``process.amain``:
    cwd / provider / model / effort / resume) plus the sidecars its session
    directory already carries. Teams, profiles, tool declarations, names and
    goals are post-open attachment state, and there is no sanctioned channel
    that carries them to a NEW child — so rather than pretending, this raises.
    The composition that DOES work, and is worth naming: open the session
    in-process, attach, dispose, and spawn against the same id — resume
    restores the attachment sidecars, and the runtime child picks them up.
    """
    blockers = [
        name
        for name, value in (
            ("agent_name", spec.agent_name),
            ("agent_id", spec.agent_id),
            ("train", spec.train),
            ("workstream", spec.workstream),
            ("team", spec.team),
            ("profile", spec.profile),
            ("tools", spec.tools),
            ("name", spec.name),
            ("goal", spec.goal),
        )
        if value not in (None, False)
    ]
    output_fields = [
        name
        for name, value in (
            ("output_format", spec.output_format),
            ("output_schema", spec.output_schema),
            ("output_retries", spec.output_retries),
        )
        if value is not None
    ]
    if output_fields:
        # A spawned runtime child is composed from its environment plus the
        # sidecars its session directory already carries; the contract is
        # post-open session state and has no sanctioned channel to a NEW child
        # (same rule as team/profile/tools/goal). The composition that works
        # is named in the remedy below and in the module docstring.
        raise SessionSpecError(
            "spawn_session() cannot carry output enforcement "
            f"({', '.join(output_fields)}); open the session with "
            "open_session(..., mode='own') and engage it afterwards"
        )
    if spec.yolo:
        blockers.append("yolo")
    if (
        spec.approvals.mode != "refuse"
        or spec.approvals.tools
        or spec.approvals.handler is not None
    ):
        blockers.append("approvals")
    if blockers:
        raise SessionSpecError(
            "spawn_session() cannot deliver post-open state to a new runtime child; "
            f"unsupported fields on this spec: {', '.join(blockers)}. Open the session "
            "in-process (open_session), attach what you need, dispose, then "
            "spawn_session() the same id — resume restores the attachment; or spawn "
            "without these fields."
        )


def _model_sample(spec: SessionSpec) -> Any:
    """The birth sample an engage carries, or ``None`` when the spec says nothing.

    ``launch._spawn_runtime`` reads a duck-typed sample (``provider``,
    ``model_id``, ``getattr(..., "reasoning_effort", None)``) — the same shape
    the desktop's viewer passes (``network/definitions.py`` calls it "the
    desktop's birth sample"). Building it here is what routes
    ``hosting``/``model``/``birth_effort`` of a NEW session into the runtime
    child: a ``WarmErrand`` is the one engagement arm that carries a sample, so
    a spec with a model warms the id first and then delivers the real errand.

    A FULL pair, or nothing — and that is the spawn contract, not a taste
    choice: ``_spawn_runtime`` writes ``initial_model.provider`` and
    ``.model_id`` to the child's environment UNCONDITIONALLY when a sample is
    present, and ``subprocess`` refuses a ``None`` value (measured: ``TypeError``
    before exec). ``spawn_session`` refuses a half pair up front so this stays
    unreachable from the public surface.
    """
    if not (spec.hosting and spec.model):
        return None
    return SimpleNamespace(
        provider=spec.hosting, model_id=spec.model, reasoning_effort=spec.birth_effort
    )


async def spawn_session(
    spec: SessionSpec,
    *,
    roots: SessionRoots,
    errand: "Errand",
    deadline_s: float | None = None,
    allow_multi_root: bool = False,
) -> "EngageOutcome":
    """Ensure a runtime exists for the spec's session and deliver ``errand``.

    Runtime-hosted, survives the caller: the runtime is a detached
    ``python -m local_operator.session.runtime.process`` child (spawned by the
    sanctioned ``engage_runtime``, never a raw ``Popen``), and the session it
    serves stays attachable by any viewer afterwards. Long-lived programmatic
    work should run under a supervisor (launchd) — the design's own note: a
    spawn from an unsupervised process inherits that process's lifetime story.

    A NEW session id is minted the way viewers mint them
    (``uuid4().hex[:12]`` — the convention ``session_factory``'s adopt path
    documents), the child ADOPTS it, and a ``WarmErrand`` carrying the spec's
    birth sample starts the runtime on the chosen provider/model before the
    real errand is delivered over the same engagement router the phone, ``lop
    send`` and the mesh use. ``spec.resume`` (or ``@latest``) skips the warm
    step: the conversation's own saved selection governs.

    The child inherits the SCOPED environment — explicit roots, ``HOME`` set to
    the agent home, ``CMUX_*``/``LOP_*`` stripped, notifications silenced
    unless opted in — because engage spawns it from this process's environment
    and there is no env parameter to pass. See :func:`_scoped_process_env`.
    """
    _guard_agent_shell()
    roots.assert_durable()
    with _live_root(roots, allow_multi_root=allow_multi_root):
        _refuse_spawn_extras(spec)

        from local_operator.resume import resolve_resume_id
        from local_operator.session.runtime.launch import WarmErrand, engage_runtime

        if spec.resume is None:
            # A HALF model pair is not routable to a new child: the birth
            # sample must carry provider and model together (see
            # ``_model_sample``), and dropping either silently would leave the
            # caller believing the child was born on their selection. Refused
            # with the remedies named.
            if (spec.hosting is None) != (spec.model is None):
                raise SessionSpecError(
                    "spawn_session() routes hosting/model to a NEW runtime as one birth "
                    "sample; state both, or neither (the child then resolves its model "
                    "from the root's config)."
                )
            if spec.birth_effort and not (spec.hosting and spec.model):
                raise SessionSpecError(
                    "birth_effort for a new spawned session rides the birth sample and "
                    "needs the hosting/model pair alongside it; state all three, or "
                    "open the session in-process (open_session) to set a birth level "
                    "without a model pair."
                )
            session_id = uuid.uuid4().hex[:12]
        else:
            # '@latest' resolves here, against THIS root, so the id the child
            # adopts is a concrete directory name — never a sentinel.
            session_id = resolve_resume_id(roots.config_path, spec.resume)

        kwargs: dict[str, Any] = {"config_dir": roots.config_path}
        if deadline_s is not None:
            kwargs["deadline_s"] = deadline_s

        with _scoped_process_env(roots, for_child=True, notifications=spec.notifications):
            _check_resolved_roots(roots)
            if spec.resume is None:
                sample = _model_sample(spec)
                await engage_runtime(
                    session_id,
                    str(roots.cwd_path),
                    WarmErrand(
                        initial_model=sample,
                        model_selection_override=bool(spec.hosting or spec.model),
                    ),
                    **kwargs,
                )
            outcome = await engage_runtime(session_id, str(roots.cwd_path), errand, **kwargs)
        return outcome


async def deliver(
    session_id: str,
    *,
    roots: SessionRoots,
    errand: "Errand",
    deadline_s: float | None = None,
    allow_multi_root: bool = False,
) -> "EngageOutcome":
    """Deliver one errand to ``session_id``, ensuring its runtime exists.

    The thin half of :func:`spawn_session`: the same engagement router, no spec
    — for a session that already exists (id or ``@latest``), including cold
    ones whose runtime the engage will start. Carries no birth sample, so a
    cold runtime it starts resolves its model the way the children of
    ``lop exec`` do (config/selection); use ``spawn_session`` when a new
    session must be born on a specific provider/model.

    Delivery is the router's: a live record gets the work over its loopback
    socket; a cold id gets a runtime first, with the same arbitration
    (transcript lease) every front end's engage goes through.
    """
    _guard_agent_shell()
    roots.assert_durable()
    with _live_root(roots, allow_multi_root=allow_multi_root):
        from local_operator.resume import resolve_resume_id
        from local_operator.session.runtime.launch import engage_runtime

        resolved_id = resolve_resume_id(roots.config_path, session_id)
        kwargs: dict[str, Any] = {"config_dir": roots.config_path}
        if deadline_s is not None:
            kwargs["deadline_s"] = deadline_s

        with _scoped_process_env(roots, for_child=True):
            _check_resolved_roots(roots)
            outcome = await engage_runtime(resolved_id, str(roots.cwd_path), errand, **kwargs)
        return outcome


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class SessionEventStream:
    """Async iterator over a session's ``AgentEvent`` stream.

    ``subscribe()`` is synchronous and returns an unsubscribe callable; this
    adapter bridges it to ``async for`` for callers that are themselves
    coroutines. Subscribe FIRST, prompt once — the exec order
    (``exec_mode``): events emitted before the subscription are gone, because
    the session's stream is not a replay log (the transcript is).

    The internal handler is synchronous and never raises: the dispatcher's
    contract is that a handler failure must not kill the session's emit, and a
    consumer's own errors belong to the consumer — this stream delivers events,
    it does not run user code. ``aclose()`` (or leaving the ``async with`` it
    supports) unsubscribes and ends any in-flight ``__anext__``; a stream left
    open until the session is disposed is simply released with it.

    The queue is unbounded on purpose: dropping events silently is the one
    thing an event stream must not do, and a consumer that stops consuming is
    the consumer's bug — with its remedy one ``aclose()`` away.
    """

    def __init__(self, session: Any) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._unsubscribe: Callable[[], None] | None = session.subscribe(self._on_event)
        self._closed = False

    def _on_event(self, event: Any) -> None:
        # Synchronous by contract (EventHandler accepts sync handlers), and
        # defensive by contract: nothing a consumer does may surface here, and
        # nothing here may abort the session's dispatch.
        try:
            self._queue.put_nowait(event)
        except Exception:  # noqa: BLE001 — a stream must never kill the emitter
            pass

    def __aiter__(self) -> "SessionEventStream":
        return self

    async def __anext__(self) -> Any:
        event = await self._queue.get()
        if event is _STREAM_CLOSED:
            raise StopAsyncIteration
        return event

    def close(self) -> None:
        """Unsubscribe and end iteration; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._queue.put_nowait(_STREAM_CLOSED)

    async def aclose(self) -> None:
        """``close()`` for ``async with`` bodies."""
        self.close()

    async def __aenter__(self) -> "SessionEventStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self.close()


def events(session: Any) -> SessionEventStream:
    """Subscribe to ``session`` and return the async iterator over its events.

    A function rather than a constructor call so the facade can keep its
    spelled surface (``from local_operator.sdk import events``) and so the
    adapter has one home; ``printable_event`` remains the JSON projection for
    line-oriented consumers — this adapter adds no second serialization.
    """
    return SessionEventStream(session)

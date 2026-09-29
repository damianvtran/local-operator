"""Value objects for the SDK's session surface: roots, spec, approval policy.

WHAT THIS MODULE IS, AND WHY IT IS SEPARATE FROM :mod:`local_operator.sdk`.
``local_operator.sdk`` is the *doing* half of the embedding surface — it drives
``session_factory.create_session`` / ``engage_runtime`` / ``spawn_owned_session``,
the exact machinery ``lop exec``, the TUI, the desktop app, the mobile relay and
the benchmark apparatus already consume. This module is the *what* half: the
frozen value objects that surface accepts, so the facade never has to grow a
second vocabulary. Nothing here runs anything; nothing here may.

THREE CONTRACTS, EACH PINNED HERE RATHER THAN IN THE FACADE:

1. :class:`SessionRoots` — **explicit roots, no ambient default.** ``paths.py``
   resolves ``config_dir()``/``agent_home_dir()`` from the environment on every
   call, which is exactly the mechanism that let a programmatic launch resolve
   the operator's real store — the incident class recorded in
   ``docs/design/sdk-engagement.md`` (the campaign that broke ``lop secret``
   machine-wide twice). A caller of the SDK states its roots; the facade scopes
   the environment around construction and *asserts* the resolved roots.
2. :class:`SessionSpec` — mirrors ``lop exec``'s session namespace. The eight
   fields ``exec_mode._make_default_session_factory`` puts in its narrow
   namespace (``hosting``, ``model``, ``agent_name``, ``agent_id``, ``yolo``,
   ``train``, ``resume``, ``workstream``) are reproduced **field-for-field** by
   :meth:`SessionSpec.to_namespace` — a parity test in
   ``tests/unit/session/test_spec.py`` pins that against exec's literal — plus
   ``birth_effort`` (``spawn_owned_session``'s documented extra) and the
   post-open attachment selectors ``exec
   --team/--profile/--tools/--goal/--name`` carries. ``to_runner_args()`` is the
   wider adapter consumed by ``exec_startup.resolve_startup`` and
   ``exec_startup.declared_tool_inventory`` — the same helpers exec runs.
3. :class:`ApprovalPolicy` — presets mirroring exec's approval behaviours.
   ``refuse()`` is the DEFAULT, and the stand-as-approval half of a tools
   declaration is opt-in via ``declared()`` rather than derived from the
   absence of a terminal the way exec derives it: an SDK process's stdin tells
   nothing about whether a person can be asked, so the caller states it. The
   mapping table is pinned by tests; see the class docstring.

IMPORT-CHEAP CONTRACT. This module imports the standard library plus
``local_operator.paths`` only — and ``paths`` is stdlib-only by its own
docstring, deliberately so it can sit on the CLI startup path. Importing it
here keeps ONE definition of the environment names (``CONFIG_DIR_ENV``,
``AGENT_HOME_ENV``) and the default directory names, instead of a second copy
that would drift. **No engine import may be added to this module**; the facade
is the one allowed the function-local heavy imports, and
``tests/unit/test_sdk.py`` guards both halves.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal

from local_operator.paths import (
    AGENT_HOME_DIRNAME,
    AGENT_HOME_ENV,
    CONFIG_DIR_ENV,
    DEFAULT_CONFIG_DIRNAME,
)

__all__ = [
    "ApprovalPolicy",
    "SessionIsolationError",
    "SessionRoots",
    "SessionSpec",
    "SessionSpecError",
    "VolatileRootError",
    "is_volatile_root",
    "uid_home_dir",
]


class SessionSpecError(ValueError):
    """Invalid input to the SDK surface — refused before anything is built."""


class VolatileRootError(SessionSpecError):
    """A session root lives somewhere the OS may purge under a live run."""


class SessionIsolationError(SessionSpecError):
    """The environment resolved a root the caller did not declare."""


#: The formats a spec may enforce, duplicated from
#: ``output_contract.OUTPUT_FORMATS`` rather than imported: this module is the
#: import-cheap half of the SDK (stdlib + ``paths``, pinned by
#: ``test_importing_spec_leaves_the_engine_off_the_graph``), and constructing a
#: spec must not drag the contract's pydantic/yaml machinery in — the same
#: stdlib-only discipline the ``tools``/``yolo`` checks follow.
#: ``tests/unit/session/test_spec.py`` pins the two tuples together, so the
#: duplication cannot drift.
_OUTPUT_FORMATS: tuple[str, ...] = ("markdown", "json", "yaml", "toml")


# ---------------------------------------------------------------------------
# Roots — explicit, durable, and checked against the resolved environment
# ---------------------------------------------------------------------------


#: Commands run with a "home" whose identity the process cannot see are how a
#: programmatic launch ends up in the operator's real store. ``Path.home()``
#: reads ``$HOME``, so an isolated run would compare its root against its own
#: redirected home, conclude it is the default, and reuse real paths; the
#: browser bridge learned this the hard way (``AGENTS.md`` §Isolating a run).
#: Every "is this the default?" question in this module therefore goes through
#: :func:`uid_home_dir`, which asks the uid's passwd entry first.
def uid_home_dir() -> Path:
    """The uid's home directory as the ACCOUNT records it, not as ``$HOME`` says.

    ``Path.home()`` follows the environment, and the whole point of an isolated
    run is that its environment lies about where home is. The passwd entry is
    the one answer an isolated child cannot redirect, so it is what "the
    operator's real root" is compared against. Falls back to ``expanduser``
    where the platform has no passwd database (Windows), which restores the
    environment-reading behaviour rather than failing there.
    """
    try:
        import pwd
    except ImportError:  # pragma: no cover — Windows
        return Path(os.path.expanduser("~"))
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError):  # pragma: no cover — a uid with no passwd entry
        return Path(os.path.expanduser("~"))


def volatile_roots() -> tuple[Path, ...]:
    """Directories the OS may purge under a live run, resolved.

    The same rule — and the same reasoning — as
    ``local_operator/evaluation/runner/durable_root.py``, the helper the
    benchmark's adapter build and episode scripts share: macOS purges
    ``/private/tmp`` on disk pressure and on a periodic sweep with no warning
    and no regard for open handles, and ``$TMPDIR`` on macOS is a per-user
    directory under ``/var/folders`` the same sweep covers. Resolved so
    ``/tmp -> /private/tmp`` cannot slip past a prefix check on the spelled
    path. Kept as a separate small function rather than an import from the
    runner because the runner tree is the evaluation apparatus — the SDK must
    not depend on it — and the rule is three lines.
    """
    roots = [Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"), Path("/private/var/tmp")]
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        roots.append(Path(tmpdir))
    out: list[Path] = []
    for root in roots:
        try:
            out.append(root.resolve())
        except OSError:  # pragma: no cover — a path nothing can stat
            out.append(root)
    return tuple(out)


def is_volatile_root(path: Path) -> bool:
    """Whether ``path`` resolves under a directory the OS may purge mid-run."""
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover
        resolved = path
    return any(resolved == root or root in resolved.parents for root in volatile_roots())


@dataclass(frozen=True, slots=True)
class SessionRoots:
    """The three roots a programmatic session runs against. All required.

    ``config_dir`` is the private store (transcripts, credentials, secret
    store); ``agent_home`` is the workspace home the agent reads and writes
    during a task; ``cwd`` is where the session runs. They mirror the
    independent overrides ``paths.config_dir()`` / ``paths.agent_home_dir()``
    honour (``LOCAL_OPERATOR_CONFIG_DIR`` / ``LOCAL_OPERATOR_HOME``), which is
    why all three are separate fields rather than one.

    There is no default and no ambient fallback. ``allow_ambient=True`` is the
    deliberate, greppable opt-out for single-machine scripts that mean to touch
    the operator's own store: it is what permits the uid-default roots and a
    uid-default cache to pass the facade's isolation checks.

    ``allow_volatile=True`` is the second opt-out, for roots under ``/tmp`` /
    ``$TMPDIR``-style directories: a run whose store lives there can have it
    purged mid-run (the benchmark's rescue-root incident), so the default
    refuses; tests and throwaway rigs that genuinely do not outlive the process
    say so explicitly.
    """

    config_dir: str | Path
    agent_home: str | Path
    cwd: str | Path
    allow_volatile: bool = False
    allow_ambient: bool = False

    def __post_init__(self) -> None:
        for name in ("config_dir", "agent_home", "cwd"):
            value = getattr(self, name)
            if not str(value).strip():
                raise SessionSpecError(f"SessionRoots.{name} must be a non-empty path")

    @staticmethod
    def _resolved(value: str | Path) -> Path:
        # ``expanduser`` so a caller may write ``~`` in a root; ``resolve`` so
        # every comparison in the facade is between real paths — the /tmp symlink
        # case is exactly why the durable rule compares resolved.
        return Path(value).expanduser().resolve()

    @property
    def config_path(self) -> Path:
        """``config_dir`` as a resolved absolute path."""
        return self._resolved(self.config_dir)

    @property
    def agent_home_path(self) -> Path:
        """``agent_home`` as a resolved absolute path."""
        return self._resolved(self.agent_home)

    @property
    def cwd_path(self) -> Path:
        """``cwd`` as a resolved absolute path."""
        return self._resolved(self.cwd)

    @property
    def default_config_path(self) -> Path:
        """The uid-default config dir this machine would use with no override."""
        return (uid_home_dir() / DEFAULT_CONFIG_DIRNAME).resolve()

    @property
    def default_agent_home_path(self) -> Path:
        """The uid-default agent home this machine would use with no override."""
        return (uid_home_dir() / AGENT_HOME_DIRNAME).resolve()

    def to_env(self) -> dict[str, str]:
        """The environment this root implies, for a child to be launched with."""
        return {
            CONFIG_DIR_ENV: str(self.config_path),
            AGENT_HOME_ENV: str(self.agent_home_path),
        }

    def same_as(self, other: SessionRoots) -> bool:
        """Whether two roots name the same store (config dir is the identity)."""
        return self.config_path == other.config_path

    def assert_durable(self) -> None:
        """Refuse a store the OS may purge mid-run, unless explicitly allowed.

        Applied by the facade to every entry point. ``cwd`` is deliberately NOT
        checked: a working directory may legitimately be anywhere the caller
        runs the session, while the two store roots hold everything a later
        reader needs — and the incident this rule comes from (a purged rescue
        root) is a property of the store, not of the workspace.
        """
        if self.allow_volatile:
            return
        for label, path in (
            ("config_dir", self.config_path),
            ("agent_home", self.agent_home_path),
        ):
            if is_volatile_root(path):
                raise VolatileRootError(
                    f"SessionRoots.{label} {path} resolves under a directory the OS may "
                    "purge mid-run (/tmp, /var/tmp or $TMPDIR); use a durable location, "
                    "or pass SessionRoots(..., allow_volatile=True) if this run "
                    "genuinely does not outlive the process"
                )


# ---------------------------------------------------------------------------
# Approval policy — the pinned mapping table
# ---------------------------------------------------------------------------

#: A host approval gate, in the two shapes ``harness/approval.py`` accepts.
_GateCallable = (
    Callable[[str, str], Awaitable[bool]] | Callable[[str, str, "str | None"], Awaitable[bool]]
)


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """How a session opened through the SDK answers tool-approval gates.

    The mapping table, pinned by tests (``tests/unit/session/test_spec.py``),
    and how each preset relates to exec:

    ==================  ==========================================  =================
    preset              gate behaviour                              exec analogue
    ==================  ==========================================  =================
    ``refuse()``        typed refusal (``ApprovalUnavailableError``)  default, non-tty
    ``auto()``          every gate call approved                    ``--yolo``
    ``declared(tools)`` the named tools stand as their own approval, ``--tools`` (piped)
                        inside a reach bound to exactly them
    ``callback(fn)``    ``fn`` decides, exactly as a front end's     a full-screen front end
                        ``set_approval_handler`` does
    ==================  ==========================================  =================

    WHERE THIS DIVERGES FROM EXEC, AND WHY. exec derives the declared-tools
    stand-as-approval from a *terminal*: ``not control and not stdin.isatty()``.
    An SDK process has no terminal to consult and its stdin says nothing about
    whether a person can be asked, so the same stand is opt-in here — the
    caller says ``declared(tools)`` — while ``refuse()`` keeps a declared
    inventory's write/exec calls gated. The effect is that the SDK default is
    strictly more conservative than a piped exec run with the same ``--tools``;
    nothing else in the mapping moves.

    ``ask`` questions (the AskUserFn surface) are NOT installed by any preset.
    That mirrors exec — none of its headless paths installs one — and it is why
    the ``ask`` tool is simply absent from an SDK session's inventory until a
    caller supplies a surface via the session object itself.
    """

    mode: Literal["refuse", "auto", "declared", "callback"]
    tools: tuple[str, ...] = ()
    handler: _GateCallable | None = None

    def __post_init__(self) -> None:
        if self.mode == "declared":
            if not self.tools:
                raise SessionSpecError(
                    "ApprovalPolicy.declared() must name at least one tool; use "
                    "ApprovalPolicy.refuse() for an undeclared bound session"
                )
        elif self.tools:
            raise SessionSpecError(
                f"ApprovalPolicy.tools is only meaningful for mode 'declared', not {self.mode!r}"
            )
        if self.mode == "callback":
            if not callable(self.handler):
                raise SessionSpecError("ApprovalPolicy.callback() requires a callable gate")
        elif self.handler is not None:
            raise SessionSpecError(
                f"ApprovalPolicy.handler is only meaningful for mode 'callback', not {self.mode!r}"
            )

    @classmethod
    def refuse(cls) -> ApprovalPolicy:
        """The default: every gated call gets a typed refusal, never a silent False."""
        return cls(mode="refuse")

    @classmethod
    def auto(cls) -> ApprovalPolicy:
        """Approve every gated call — the same posture as ``exec --yolo``."""
        return cls(mode="auto")

    @classmethod
    def declared(cls, tools: Sequence[str]) -> ApprovalPolicy:
        """Bound the reach to ``tools`` and let the declaration stand as approval.

        The reach half is enforced by ``Session.set_tool_inventory`` — excluded
        tools are unreachable by name, not merely unapproved — and the approval
        half makes the listed names answer their own gate, exactly like a
        non-trailing ``exec --tools`` declaration where nobody can be asked.
        """
        names = tuple(dict.fromkeys(str(name).strip() for name in tools if str(name).strip()))
        return cls(mode="declared", tools=names)

    @classmethod
    def callback(cls, handler: _GateCallable) -> ApprovalPolicy:
        """Install ``handler`` via ``set_approval_handler`` — a front end's own gate."""
        return cls(mode="callback", handler=handler)

    @property
    def stands_as_approval(self) -> bool:
        """Whether the inventory declaration answers its own members' gates."""
        return self.mode == "declared"


# ---------------------------------------------------------------------------
# SessionSpec — the exec-parity session description
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """What to open: the exec-parity session fields plus post-open attachments.

    The first eight fields are **pinned to exec's narrow namespace** — a parity
    test builds exec's namespace through ``exec_mode._make_default_session_factory``
    and asserts field-for-field equality with :meth:`to_namespace` — and the
    rest mirror the selectors ``lop exec`` applies after construction
    (``--team``, ``--profile``, ``--tools``, ``--goal``, ``--name``), plus the
    two SDK-owned decisions: ``approvals`` (see :class:`ApprovalPolicy`) and
    ``notifications``.

    ``notifications`` is the "nobody is watching" default made explicit: a
    *spawned* runtime launched from an SDK spec is silenced (the same
    ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` switch the harness sets for its own
    children) unless the caller opts in with ``notifications=True``. In-process
    sessions leave notification policy to the host process.

    Deliberately absent in v1, and where each lives instead: ``--effort`` is a
    runner-level knob applied by ``exec_session`` after construction (use
    ``birth_effort`` for the construction-time level); ``--loop``/``--loop-goal``
    are exec's continuation mechanism and an SDK caller drives turns itself;
    ``--clear-goal`` adjusts a resumed run rather than starting one.
    """

    hosting: str | None = None
    model: str | None = None
    agent_name: str | None = None
    agent_id: str | None = None
    yolo: bool = False
    train: bool = False
    #: Session id to resume, or ``@latest``. ``""`` is refused (exec parity:
    #: ``--resume ""`` is a user error, and starting a NEW session under it
    #: would read as a resume that lost its history).
    resume: str | None = None
    #: Ask for this run as a long-lived parallel workstream the operator wants
    #: listed (exec ``--workstream``). Meaningful only under an agent's shell;
    #: the stamp itself is written by ``session_factory._prepare``, the same
    #: place exec's stamp is written, so the SDK cannot drift from it.
    workstream: bool = False
    #: Construction-time reasoning level (``spawn_owned_session``'s documented
    #: extra; clamped by the factory rather than refused).
    birth_effort: str | None = None
    team: str | None = None
    profile: str | None = None
    #: The run's whole reach when set — exec ``--tools`` semantics, a one-way
    #: declaration for the session's life. ``None`` leaves the session
    #: unrestricted. An empty sequence is refused at preflight, like exec.
    tools: tuple[str, ...] | None = None
    approvals: ApprovalPolicy = field(default_factory=ApprovalPolicy.refuse)
    name: str | None = None
    goal: str | None = None
    #: False (default) silences a *spawned* runtime; True lets it inherit the
    #: launcher's notification posture. In-process sessions ignore this field.
    notifications: bool = False
    #: Enforce the format of the assistant's final response for every turn of
    #: this session: "markdown" | "json" | "yaml" | "toml". None (default)
    #: leaves every turn byte-identical to today's behaviour.
    output_format: str | None = None
    #: Schema the decoded payload must validate against (json/yaml/toml), or a
    #: MarkdownSchema / {"required_sections": [...]} for markdown. Accepts a
    #: pydantic model class, dataclass, TypedDict — anything
    #: ``pydantic.TypeAdapter`` accepts — or a raw JSON Schema mapping.
    #: Validation is ``OutputContract``'s, so ``output_schema`` without
    #: ``output_format`` is refused below rather than silently inert.
    output_schema: object | None = None
    #: Max retries after a rejected final response (0-5; default 2).
    output_retries: int | None = None

    def __post_init__(self) -> None:
        if self.agent_name and self.agent_id:
            # exec refuses this pair at resolve_startup; refusing here too keeps
            # the failure before any construction, with the same wording.
            raise SessionSpecError("agent_name and agent_id are mutually exclusive")
        if self.resume is not None and not str(self.resume):
            raise SessionSpecError('resume must be a session id or "@latest", not an empty string')
        if self.tools is not None and not isinstance(self.tools, tuple):
            # A list is the ergonomic spelling; frozen here so the spec is
            # hashable and cannot be mutated between validation and use.
            object.__setattr__(self, "tools", tuple(self.tools))
        if self.tools is not None and not self.tools:
            raise SessionSpecError(
                "tools must name at least one tool; leave it None to stay unrestricted"
            )
        if self.yolo and self.approvals.mode in ("refuse", "callback"):
            # yolo short-circuits the gate object on the session side, so a
            # refuse/callback gate installed here could never be consulted —
            # silently ignored is exactly the failure shape this SDK refuses to
            # have. `auto()` is the honest spelling of the same posture.
            raise SessionSpecError(
                f"yolo=True makes an ApprovalPolicy.{self.approvals.mode}() gate "
                "unreachable; use ApprovalPolicy.auto() (or drop yolo) instead"
            )
        # The output-contract fields are validated cheaply HERE (stdlib only —
        # see ``_OUTPUT_FORMATS``) and fully by ``OutputContract.__post_init__``
        # when the session is built, which both the SDK and exec run; these
        # refusals exist so a spec that cannot possibly work fails where it is
        # written, before anything is constructed.
        if self.output_format is not None:
            if self.output_format not in _OUTPUT_FORMATS:
                raise SessionSpecError("output_format must be one of " + ", ".join(_OUTPUT_FORMATS))
        if self.output_schema is not None and self.output_format is None:
            raise SessionSpecError("output_schema requires output_format")
        if self.output_retries is not None and self.output_format is None:
            raise SessionSpecError("output_retries requires output_format")
        if self.output_retries is not None and (
            isinstance(self.output_retries, bool)
            or not isinstance(self.output_retries, int)
            or not 0 <= self.output_retries <= 5
        ):
            raise SessionSpecError("output_retries must be between 0 and 5")

    def to_namespace(self) -> argparse.Namespace:
        """The narrow session-factory namespace, field-for-field as exec builds it.

        Do not add fields here. ``session_factory._prepare`` receives only this
        namespace, and the parity test pins it to the literal in
        ``exec_mode._make_default_session_factory``; anything runner-level
        belongs in :meth:`to_runner_args`. The output-contract fields are
        deliberately absent for that same reason (the eight-field pin is the
        published adapter's shape), and they need no home here anyway: they are
        applied post-open through ``Session.set_output_contract``, exactly like
        ``tools``/``goal``/``name``.
        """
        return argparse.Namespace(
            hosting=self.hosting,
            model=self.model,
            agent_name=self.agent_name,
            agent_id=self.agent_id,
            yolo=self.yolo,
            train=self.train,
            resume=self.resume,
            workstream=self.workstream,
        )

    def to_runner_args(self) -> argparse.Namespace:
        """The wider adapter the exec helpers consume (``resolve_startup`` etc.).

        This is the SDK's half of the Namespace coupling risk named in the
        design (§7.1): ``create_session`` consumes an ``argparse.Namespace``,
        and the adapter is the only sanctioned producer besides exec. The
        fields are the session fields plus every selector
        ``exec_startup.resolve_startup`` / ``apply_startup`` /
        ``declared_tool_inventory`` read — ``tools`` in exec's own comma-string
        form — with the runner-level keys exec's runner owns (``loop``,
        ``clear_goal``, ``control``) present and off, so the helpers' ``getattr``
        reads are answered rather than defaulted.

        The output-contract keys are deliberately NOT included: the SDK's schema
        spelling is a type or mapping, not a file path, so the shared
        ``resolve_startup`` validation (which reads a path) does not cover it.
        ``resolve_output_contract`` reads the three keys with ``getattr(...,
        None)`` defaults and an absent key means "not declared"; the common
        validation is ``OutputContract.__post_init__``, which
        ``sdk._build_session`` runs directly.
        """
        return argparse.Namespace(
            **vars(self.to_namespace()),
            birth_effort=self.birth_effort,
            profile=self.profile,
            # exec stores --tools as the comma-separated string; the helpers
            # parse it through exec_startup.parse_tool_inventory.
            tools=",".join(self.tools) if self.tools is not None else None,
            goal=self.goal,
            name=self.name,
            clear_goal=False,
            loop=None,
            loop_goal=None,
            control=False,
        )

    def with_resume(self, session_id: str) -> SessionSpec:
        """A copy of this spec that resumes ``session_id``.

        The design sketch spells this ``spec.resume(id)``; the field that must
        stay named ``resume`` for exec parity takes the attribute, so the
        helper name moves instead — same call, different spelling.
        """
        if not session_id:
            raise SessionSpecError("with_resume() needs a session id")
        return replace(self, resume=session_id)

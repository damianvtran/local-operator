"""The ``/info`` snapshot's dataclasses. Pure data: no I/O, no rich, no Textual.

Split from :mod:`local_operator.info.collect` for the reason
``local_operator/analytics/`` splits ``model.py`` from ``store.py``, and for one
harder constraint on top of it. ``collect`` has to reach ``mobile.resources``,
``browser_bridge.state``, ``credentials``, ``agents`` and ``teams`` — none of
which belong anywhere near a ``lop --version`` — while this module is what a
future ``lop info`` renderer and every unit test import. Keeping the two apart
makes "the shapes are cheap and the probes are not" a property of the import
graph rather than of anyone's discipline, and ``test_model.py`` asserts it.

**Every field has a default, and an all-defaults instance must render.** That is
not tidiness: ``/info`` is the screen a user opens when something is already
broken, so the collector degrades field by field (see :func:`collect._safe`) and
hands the renderer whatever it managed to read. A dataclass with a required
field would turn one unreadable probe into an empty screen, which is the exact
failure this design exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Sentinel a renderer prints for a scalar nobody could read. Matches what
#: ``lop sessions`` already prints for an unmeasurable pid
#: (``mobile/resources.py``: "the caller prints ``—`` for unknowns"), so the two
#: surfaces spell "we looked and could not tell" the same way.
UNKNOWN = "—"


@dataclass(frozen=True)
class InstallInfo:
    """Which build this is, where it came from, and whether it is current.

    The editable-vs-uv-tool distinction is the single most common source of "I
    fixed it and nothing changed" in this codebase (AGENTS.md devotes a whole
    section to it), so ``kind`` and ``prefix`` are as load-bearing as
    ``version``.
    """

    version: str = ""
    #: ``update.InstallKind`` value, as a plain string — this module must not
    #: import :mod:`local_operator.update` (see the module docstring).
    kind: str = ""
    prefix: str = ""
    executable: str = ""
    #: The package directory ``local_operator`` ACTUALLY resolved to, from
    #: ``local_operator.__file__``. Reported alongside :attr:`prefix` rather
    #: than assumed equal to it, because the two genuinely come apart and the
    #: divergence is invisible from every other surface:
    #:
    #: ``session/runtime/launch.py:287`` spawns the runtime as
    #: ``[sys.executable, "-m", ...]`` with NO ``cwd=``, so the launching
    #: session's working directory lands on ``sys.path[0]`` ahead of
    #: site-packages. A session whose cwd is a checkout of this repo therefore
    #: runs THAT CHECKOUT's code while every other field — version, install
    #: kind, prefix — describes the install it was launched from, and ``/reload``
    #: does not fix it. AGENTS.md documents the same class of trap for
    #: symlinked venvs: "an agent that tested the feature interactively this way
    #: has tested nothing, and will report a working change as broken — or
    #: worse, a broken change as working."
    #:
    #: This is the field that makes "which code am I actually running"
    #: answerable without inference, which is the whole reason the screen
    #: exists.
    import_path: str = ""
    #: True when :attr:`import_path` does not sit under :attr:`prefix`. A
    #: WARNING state, not an unavailable one: everything was read successfully,
    #: and what it says is that this runtime is executing code from somewhere
    #: other than the install it reports.
    #:
    #: Do NOT read this field directly to decide whether to warn — call
    #: :func:`is_shadowed_install`, which also excludes the benign editable
    #: case. QA round 1 (Q1) found the screen and the export reaching opposite
    #: conclusions from one snapshot because each spelled the condition itself.
    import_path_foreign: bool = False
    is_git_snapshot: bool = False
    source_ref: str = ""
    build_age_s: float | None = None
    #: The last PyPI answer already on disk, from ``update.cached_latest`` — a
    #: read that NEVER fetches. ``None`` means "never checked", which is a
    #: different fact from "up to date" and must render differently.
    latest_known: str | None = None
    latest_age_s: float | None = None
    behind: bool = False
    python_version: str = ""
    python_implementation: str = ""
    platform: str = ""
    machine: str = ""


@dataclass(frozen=True)
class ProcessInfo:
    """The runtime the user is typing into, and the directories it resolved.

    ``config_dir`` and ``cache_dir`` are carried side by side deliberately: the
    cache root derives from ``$HOME`` independently of
    ``LOCAL_OPERATOR_CONFIG_DIR``, which AGENTS.md documents as silently
    producing a *plausible wrong answer* in an "isolated" run. Showing both is
    how a reader sees the divergence instead of assuming it away.
    """

    pid: int = 0
    session_id: str = ""
    conversation_name: str = ""
    cwd: str = ""
    model_label: str = ""
    effective_model: str = ""
    uptime_s: float | None = None
    config_dir: str = ""
    config_dir_redirected: bool = False
    agent_home: str = ""
    agent_home_redirected: bool = False
    cache_dir: str = ""
    log_dir: str = ""
    #: Control-socket PORT only. The record's ``control_key`` is never carried
    #: anywhere in this module — see :class:`SessionLine`.
    control_port: int | None = None
    protocol: int | None = None
    kind: str = ""


@dataclass(frozen=True)
class SessionLine:
    """One session on this machine, as ``lop sessions`` already describes it.

    **``control_key`` is deliberately absent, and its absence is a tested
    invariant** (``tests/unit/info/test_redaction.py``). ``SessionRecord``
    carries a 64-hex control key that is the entire authorization story of the
    control socket, and ``/info``'s output is what lands in a GitHub issue.
    Omitting the field from the dataclass — rather than omitting it from one
    renderer — is what makes the guarantee survive a later renderer change:
    there is no attribute for a future ``asdict``-style dump to reach.
    """

    pid: int = 0
    kind: str = ""
    #: ``live`` | ``wedged`` | ``stale``. ``stale`` means the scan just DELETED
    #: the record file (``registry.scan``), not that the session is idle.
    state: str = ""
    session_id: str = ""
    conversation_name: str = ""
    model_label: str = ""
    cwd: str = ""
    uptime_s: float = 0.0
    heartbeat_age_s: float = 0.0
    rss_bytes: int | None = None
    footprint_bytes: int | None = None
    pending: str | None = None
    busy: bool = False
    detached: bool = False
    version: str = ""
    source_ref: str = ""
    #: Whether this row is the session running ``/info``. The reason the row
    #: builder was EXTRACTED rather than copied: ``lop sessions`` has no such
    #: concept, and a second implementation would have drifted on the subtler
    #: fields (the ``getattr`` defaulting for mid-upgrade records) long before
    #: it drifted on this one.
    is_self: bool = False


@dataclass(frozen=True)
class SessionsInfo:
    """Every session on this machine, plus the roll-ups the header reads from."""

    lines: tuple[SessionLine, ...] = ()
    total: int = 0
    live: int = 0
    wedged: int = 0
    stale: int = 0
    busy: int = 0
    pending: int = 0
    detached: int = 0
    #: More than one distinct non-empty ``(version, source_ref)`` among LIVE
    #: sessions: a host mid-upgrade, running two builds at once. Worth naming,
    #: because "I fixed it and the other window still does it" is that.
    build_skew: bool = False
    #: False when the memory probes returned nothing at all, so the screen can
    #: say "not measured" instead of printing a column of ``—`` that reads as
    #: every session using no memory.
    usage_available: bool = True
    #: False when the scan itself failed, which replaces the whole section
    #: header rather than annotating it.
    available: bool = True


@dataclass(frozen=True)
class SubagentLine:
    """One node of this session's subagent lineage, flattened depth-first."""

    job_id: str = ""
    label: str = ""
    #: ``running`` | ``queued`` | ``starting`` | ``pausing`` | ``paused`` |
    #: ``completed`` | ``failed`` | ``cancelled`` | ``gone``.
    status: str = ""
    depth: int = 0
    agent_role: str = ""
    effort: str = ""
    parent_job_id: str | None = None
    session_id: str | None = None
    live: bool = False


@dataclass(frozen=True)
class AgentsInfo:
    """Agent profiles, teams, and the subagent tree of THIS session.

    The tree is this session's only. ``SubagentComms`` is a live in-memory
    object on one ``Session`` and ``SessionRecord`` carries no subagent field,
    so nothing about another session's children is observable without a new
    control-socket op and a ``PROTOCOL_VERSION`` bump. The screen says so rather
    than implying a fleet-wide view; :attr:`cross_session_known` is the seam a
    later change would flip, and is always ``False`` today.
    """

    profiles: int = 0
    teams: int = 0
    tree: tuple[SubagentLine, ...] = ()
    running: int = 0
    queued: int = 0
    #: Settled AND STILL RETAINED. ``SubagentComms._evict_overflow`` drops
    #: settled records past its cap, so this is not "settled ever" and the label
    #: must not claim it is — a count that silently under-reports inside a bug
    #: report is worse than one that states what it counts.
    settled: int = 0
    max_running: int | None = None
    at_capacity: bool = False
    max_depth: int = 0
    #: Number of nodes below the render depth cap, folded into one summary row
    #: rather than indented off the card.
    deeper: int = 0
    cross_session_known: bool = False


@dataclass(frozen=True)
class EnvInfo:
    """The debugging extras a bug report is asked for on the second round trip.

    Credential KEY NAMES only — never a value, a length, a prefix or a hash. A
    "first four characters" habit is how key prefixes end up in issues, and the
    diagnostic question ("is the key even set?") is fully answered by the name.

    **No tool-call or request success/failure RATE belongs here, deliberately.**
    ``/session`` owns call-outcome measurement (tool-call validity against
    execution errors, each with a stated denominator), and a second
    success-ish figure computed here would disagree with it in cases neither
    screen explains — leaving a user with two numbers and no way to tell which
    is right. The MCP counts below are a DIFFERENT quantity and stay: server
    reachability is not call outcome. The surface split this preserves is
    ``/analytics`` = historical spend across sessions, ``/session`` = this
    session's ledger and health, ``/info`` = install provenance and what is
    actually running.
    """

    mcp_configured: int = 0
    mcp_connected: int = 0
    mcp_failed: int = 0
    #: True while servers deferred past the startup gate are still connecting.
    #: Shown, because reporting "1 of 3 up" mid-handshake makes a user file a
    #: bug about a server that came up a second later.
    mcp_settling: bool = False
    mcp_failures: tuple[tuple[str, str], ...] = ()
    approval_mode: str = ""
    theme: str = ""
    terminal_size: tuple[int, int] | None = None
    term: str = ""
    colorterm: str = ""
    multiplexer: str = ""
    is_tty: bool = False
    #: ``extension`` | ``extension (stale)`` | ``none``, from the FILE
    #: classification (``browser_bridge.state.liveness``) — 0.06 ms and unable
    #: to hang. Availability nuance is ``/browser``'s job, not a diagnostic's.
    browser_backend: str = ""
    browser_name: str = ""
    browser_paired: bool = False
    mobile_installed: bool = False
    mobile_healthy: bool = False
    #: ``None`` when the relay is not installed. A port is only a measurement
    #: when something is listening on it; publishing the module's DEFAULT_PORT
    #: regardless made an unconfigured host report a plausible number, which is
    #: the "absent is not a measured value" rule this screen inherits from
    #: ``/session`` (review round 1, N2).
    mobile_port: int | None = None
    credential_keys: tuple[str, ...] = ()
    guides: int = 0
    skills: int = 0


def format_duration(seconds: float | None) -> str:
    """``42s`` / ``4m`` / ``3h 12m`` / ``2d 4h``. ONE spelling for both surfaces.

    Lives here, beside the dataclasses, because both the screen and the export
    render the same fields and a quantity that reads two ways across two halves
    of one screen is a defect — this PR's own B2/Q1 finding was exactly that,
    two surfaces reaching opposite conclusions from one snapshot. The two copies
    happened to agree on every sampled value when they were measured, which is
    the argument FOR sharing rather than against it: they agree today and
    nothing made them keep agreeing.

    ``model.py`` is the right home because it is the module both already import
    and it is stdlib-only by contract, so sharing costs no new coupling.

    ``None`` is "not measured" and says so; it is never a zero duration.
    """
    if seconds is None:
        return "unknown"
    total = int(max(0.0, seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h {(total % 3600) // 60}m"
    return f"{total // 86400}d {(total % 86400) // 3600}h"


def format_bytes(value: int | None) -> str:
    """``181 MB`` / ``1.9 GB``, or :data:`UNKNOWN` when nothing was measured.

    Shared for the same reason as :func:`format_duration`. ``None`` renders as
    the unknown sentinel rather than ``0 MB``: a process whose memory could not
    be read is not a process using no memory.
    """
    if value is None:
        return UNKNOWN
    if value >= 1 << 30:
        return f"{value / (1 << 30):.1f} GB"
    return f"{value / (1 << 20):.0f} MB"


def is_shadowed_install(install: InstallInfo) -> bool:
    """Whether the running code is a checkout SHADOWING the reported install.

    **The single source of truth for that judgement**, called by the screen and
    by the export. Both used to spell the condition themselves, and they drifted
    exactly as you would expect: the screen excluded the editable case and the
    export did not, so every contributor running an editable checkout — which is
    every contributor to this repo — copied a bug report asserting their install
    was shadowed, sending a maintainer after AGENTS.md's most-documented trap
    while it was not occurring (QA round 1, Q1).

    An editable install's package IS the checkout by design, so a divergence
    there is expected rather than alarming. Every other kind diverging is the
    ``runtime/launch.py`` trap: spawned with ``-m`` and no ``cwd=``, so the
    launching session's directory precedes site-packages on ``sys.path``.
    """
    return install.import_path_foreign and install.kind != "editable"


@dataclass(frozen=True)
class InfoSnapshot:
    """Everything ``/info`` renders, with each block degrading independently.

    :attr:`degraded` maps a block or field name to the one-line reason it could
    not be read. A bug report has to be able to say WHY a field is ``—``:
    without the reason, an unreadable value costs the reporter a second round
    trip, which is the cost this screen exists to remove.
    """

    install: InstallInfo = field(default_factory=InstallInfo)
    process: ProcessInfo = field(default_factory=ProcessInfo)
    sessions: SessionsInfo = field(default_factory=SessionsInfo)
    agents: AgentsInfo = field(default_factory=AgentsInfo)
    env: EnvInfo = field(default_factory=EnvInfo)
    degraded: tuple[tuple[str, str], ...] = ()
    captured_at: float = 0.0

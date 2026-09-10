"""The ``/info`` probes: fill :mod:`local_operator.info.model`'s shapes from the
real system.

**Every heavyweight import here is function-local**, following ``update.py``'s
``import httpx`` inside ``_fetch_pypi_version`` and ``cli.py``'s deferred
imports, for the reason ``session/runtime/types.py`` states outright: modules on
the CLI startup path are paid for by every ``lop`` invocation including
``--version``. This module reaches ``mobile.resources``, ``browser_bridge``,
``credentials``, ``agents`` and ``teams``; none of that may become importable
cost for anyone who is not looking at the screen.

**Two collection phases, and the split is load-bearing.**

:func:`collect_live` reads only in-memory state (the subagent graph, the job
manager's capacity, the theme, the terminal size) and runs ON the event loop,
before the screen yields. :func:`collect_snapshot` does every blocking probe and
runs in a worker thread. The reason is the same rule
``SessionDiagnostics.capture`` states at ``session_panel.py``: a ``/new`` or
``/resume`` landing during the disk read must not put a NEW subagent tree under
an OLD header. The blocking half is genuinely slow —
``session_resource_usage`` measured **879.5 ms** for 12 pids on this host,
because on macOS it shells ``top -l1`` for the whole system — which is 26
dropped frames at 30 fps, so it cannot be anywhere near the paint path.

**No probe here may reach the network.** ``update.check_latest()`` is banned on
this path and :func:`collect_install` calls ``update.cached_latest()`` instead;
``tests/unit/info/test_collect.py`` pins that with a raising fake rather than a
timing bound. See ``cached_latest``'s own docstring for the measurement.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

from local_operator.info.model import (
    AgentsInfo,
    EnvInfo,
    InfoSnapshot,
    InstallInfo,
    ProcessInfo,
    SessionLine,
    SessionsInfo,
    SubagentLine,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: How deep the tree is walked before the remainder is folded into a summary
#: row. Matches the renderer's cap: nesting past three is rare, and at that
#: point the COUNT is the fact that matters rather than the indentation.
MAX_TREE_DEPTH = 3

#: Environment markers that identify the terminal multiplexer, most specific
#: first. PRESENCE only — a marker's VALUE can carry a socket path or a
#: workspace id, both identifying, and this screen is pasted into issues.
#:
#: Spelled as module-level constants rather than as string literals inside the
#: tuple because ``tests/unit/test_ambient_env_isolation.py`` resolves env reads
#: through the AST: it follows a ``NAME = "VAR"`` constant but cannot see a name
#: buried in a tuple-of-tuples literal. Written the other way, three of these
#: reads were invisible to the audit that exists to catch exactly that, so the
#: names went unaccounted while looking accounted for (QA round 1, Q4).
_ENV_CMUX_SOCKET_PATH = "CMUX_SOCKET_PATH"
_ENV_CMUX_WORKSPACE_ID = "CMUX_WORKSPACE_ID"
_ENV_TMUX = "TMUX"
_ENV_STY = "STY"
_ENV_ZELLIJ = "ZELLIJ"
_ENV_WEZTERM_PANE = "WEZTERM_PANE"

_MULTIPLEXER_MARKERS: tuple[tuple[str, str], ...] = (
    (_ENV_CMUX_SOCKET_PATH, "cmux"),
    (_ENV_CMUX_WORKSPACE_ID, "cmux"),
    (_ENV_TMUX, "tmux"),
    (_ENV_STY, "screen"),
    (_ENV_ZELLIJ, "zellij"),
    (_ENV_WEZTERM_PANE, "wezterm"),
)


def _safe(name: str, fn: Callable[[], T], default: T, errors: list[tuple[str, str]]) -> T:
    """Run one probe; a failure becomes ``default`` plus a NAMED reason.

    A whole-snapshot ``try/except`` is what this exists to prevent. ``/info`` is
    opened when something is already wrong, so one unreadable probe must cost
    exactly its own field — a single broken read must never erase the version
    number, which is the field a bug report most needs.

    The reason is KEPT rather than swallowed the way this codebase's other
    best-effort paths swallow theirs, because a bare ``cache dir  —`` sends the
    reporter back for a second round trip to find out whether the field does not
    apply or the screen is broken.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — a diagnostic screen never raises
        logger.debug("info: %s probe failed", name, exc_info=True)
        errors.append((name, f"{type(exc).__name__}: {exc}"[:120]))
        return default


def _resolved_import_path() -> str:
    """Where ``import local_operator`` ACTUALLY came from, resolved.

    ``__file__`` is the package's ``__init__.py``; its parent is the directory a
    reader can compare against the install prefix, and it is the exact command
    AGENTS.md prescribes for settling this by hand
    (``python -c "import local_operator; print(local_operator.__file__)"``).
    Symlinks are resolved so a linked venv cannot report the path it was reached
    THROUGH rather than the tree it actually reads.
    """
    import local_operator

    return str(Path(local_operator.__file__).resolve().parent)


def _import_path_is_foreign(import_path: str, prefix: str) -> bool:
    """Whether the running code sits outside the install it claims to be.

    A containment test rather than equality: an install's package legitimately
    lives several directories under its prefix
    (``<prefix>/lib/python3.12/site-packages/local_operator``), so equality
    would flag every healthy install and this warning would mean nothing.

    An EDITABLE install is the one legitimate divergence — its package is the
    checkout by design — but it is not special-cased here, because the case this
    exists to catch is indistinguishable from it at this level and the renderer
    has ``kind`` in hand to word the two differently.
    """
    if not import_path or not prefix:
        return False
    try:
        return not Path(import_path).is_relative_to(Path(prefix).resolve())
    except (OSError, ValueError):
        # An unresolvable path is not evidence of a foreign import; claiming a
        # warning state from a failed comparison would be a false alarm on the
        # screen whose job is to be trusted.
        return False


def collect_install(errors: list[tuple[str, str]]) -> InstallInfo:
    """Which build this is and where it came from. No network, ever."""
    import platform

    from local_operator import update

    version = _safe("install.version", update.installed_version, "", errors)
    kind = _safe("install.kind", lambda: update.install_kind().value, "", errors)
    latest, latest_age = _safe(
        # NOT ``check_latest``: that is the upgrade path and its TTL miss makes a
        # live 5 s HTTP call and rewrites the cache. See ``update.cached_latest``.
        "install.latest",
        update.cached_latest,
        (None, None),
        errors,
    )
    import_path = _safe("install.import_path", _resolved_import_path, "", errors)
    return InstallInfo(
        version=version,
        kind=kind,
        prefix=sys.prefix,
        executable=sys.executable,
        import_path=import_path,
        import_path_foreign=_import_path_is_foreign(import_path, sys.prefix),
        is_git_snapshot=_safe("install.snapshot", update.is_git_snapshot, False, errors),
        source_ref=_safe("install.source_ref", update.source_ref, "", errors),
        build_age_s=_safe("install.build_age", update.build_marker_age_s, None, errors),
        latest_known=latest,
        latest_age_s=latest_age,
        behind=update.is_behind(version, latest),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        # ~15 ms, measured. CPython memoizes ``platform.uname()`` but NOT
        # ``platform.platform()``, which re-formats on every call, so this is
        # paid each time rather than once (review round 1, N4 — the earlier
        # comment here asserted a caching behaviour that does not exist). Cheap
        # either way on the worker thread.
        platform=_safe("install.platform", platform.platform, "", errors),
        machine=_safe("install.machine", platform.machine, "", errors),
    )


def own_control_port(root: Path | None = None) -> int | None:
    """This process's own control-socket port, read from its published record.

    Read HERE rather than carried on :class:`SessionLine`, which is kept to
    exactly the shape ``lop sessions`` publishes plus ``is_self``. The port
    itself is safe to render — a port number is not a credential and it is the
    useful half for debugging an attach — but the record it sits in also holds
    ``control_key``, so the narrow read is the one that cannot grow a leak by
    someone later adding "the whole record" to the session dataclass.
    """
    import json

    from local_operator.session.runtime import registry

    raw = json.loads(registry.record_path(os.getpid(), root).read_text(encoding="utf-8"))
    port = raw.get("control_port")
    return int(port) if isinstance(port, (int, float)) else None


def collect_process(
    live: "LiveState",
    *,
    self_line: SessionLine | None,
    errors: list[tuple[str, str]],
    root: Path | None = None,
) -> ProcessInfo:
    """This runtime, and the four directories it actually resolved.

    ``config_dir`` and ``cache_dir`` are both reported because they are derived
    INDEPENDENTLY — the cache root comes off ``$HOME`` whatever
    ``LOCAL_OPERATOR_CONFIG_DIR`` says — and AGENTS.md records that divergence
    as one that silently produces a plausible wrong answer. Seeing the two
    side by side is the whole diagnostic.

    ``uptime_s`` prefers this process's own published ``SessionRecord``, matched
    on pid by the scan that :func:`collect_sessions` already ran, so ``/info``
    and ``lop sessions`` cannot report two different uptimes for one session.
    """
    from local_operator import paths, update
    from local_operator.session.runtime.types import PROTOCOL_VERSION

    return ProcessInfo(
        pid=os.getpid(),
        session_id=live.session_id,
        conversation_name=live.conversation_name,
        cwd=_safe("process.cwd", os.getcwd, "", errors),
        model_label=live.model_label,
        effective_model=live.effective_model,
        uptime_s=self_line.uptime_s if self_line is not None else live.uptime_s,
        config_dir=_safe("process.config_dir", lambda: str(paths.config_dir()), "", errors),
        config_dir_redirected=bool(os.environ.get(paths.CONFIG_DIR_ENV)),
        agent_home=_safe("process.agent_home", lambda: str(paths.agent_home_dir()), "", errors),
        agent_home_redirected=bool(os.environ.get(paths.AGENT_HOME_ENV)),
        cache_dir=_safe("process.cache_dir", lambda: str(update.default_cache_dir()), "", errors),
        log_dir=_safe("process.log_dir", lambda: str(paths.log_dir()), "", errors),
        # Absent for a runtime that never published a record (exec mode, a
        # test harness): that is a legitimate state, not a failure, so it is
        # not recorded in ``degraded``.
        control_port=(
            _safe("process.control_port", lambda: own_control_port(root), None, errors)
            if self_line is not None
            else None
        ),
        protocol=PROTOCOL_VERSION,
        kind=self_line.kind if self_line is not None else live.kind,
    )


#: Above this, a published count is treated as corrupt rather than as a
#: measurement. Deliberately far above anything this codebase can produce —
#: ``DEFAULT_MAX_RUNNING_JOBS`` is 15 and the count is a ``len()`` over a
#: bounded roster — so it can only reject a foreign or damaged record, never a
#: real fleet. It is a RENDERING ceiling, not a belief about how many subagents
#: can exist: six digits still fit the narrow rung.
_ABSURD_COUNT = 999_999


def _reported_count(value: Any) -> int | None:
    """A published subagent count, or ``None`` when the record did not report one.

    ``SessionRecord.from_json`` filters keys and calls the constructor — it does
    no type validation — so every field on a record is whatever the writer put
    in the file. That is fine for the strings and bools already read here, which
    only ever get formatted, but these two are the first record fields this
    module does ARITHMETIC on, and arithmetic is where a foreign value stops
    being cosmetic:

    * a ``str`` or ``list`` raises ``TypeError`` inside the roll-up. ``_safe``
      guards whole SECTIONS, so one bad record cost the entire sessions block —
      no table, no runtimes row, and no lower-bound caveat — on a screen whose
      whole purpose is describing a host that is already broken. Before these
      fields existed there was no arithmetic here and the same record listed
      normally, so that was a regression rather than a new limitation.
    * a merely-numeric wrong value does not raise at all, which is worse: a
      float printed ``4.5 total — 1 sessions + 3.5 subagents`` and a negative
      printed ``-1 subagents``, both as measured fact.

    Anything that is not a non-negative ``int`` is therefore treated as NOT
    REPORTED rather than sanitised into a number. That is this screen's own
    contract applied one layer out: an unusable value is not a measurement, and
    calling it ``None`` folds it into the lower-bound caveat, which already
    exists to say the total is missing terms. ``bool`` is excluded explicitly —
    it is an ``int`` subclass, so ``True`` would otherwise count as one subagent.

    A count above ``_ABSURD_COUNT`` is refused the same way. It is not that the
    number is wrong — it is that a 31-digit figure renders 81 cells wide and
    overflows every frame, including the abbreviated rung that exists to serve
    narrow ones, because the shed compresses the LABELS and not the FIGURE.
    Treating it as unreported keeps a corrupt record from breaking the layout
    of the screen you open when something is already broken.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    if value > _ABSURD_COUNT:
        return None
    return value


def collect_sessions(
    root: Path | None = None,
    *,
    scan: Callable[..., list[tuple[Any, str]]] | None = None,
    usage: Callable[..., dict[int, Any]] | None = None,
    self_pid: int | None = None,
    now: float | None = None,
    include_stored: bool = False,
    stored_limit: int | None = None,
) -> SessionsInfo:
    """Every session on this machine, as ``lop sessions`` already describes them.

    **This is the ONE implementation.** ``cli.sessions_command`` calls it for
    both its table and its ``--json`` output; a second copy of "which sessions
    exist and what do they cost" is precisely the drift this codebase writes
    essays about avoiding. Two specific details make the duplication dangerous
    rather than merely untidy:

    * the ``getattr(rec, ..., default)`` spellings below exist because a record
      written by an OLDER runtime lacks those fields entirely. That rule has to
      hold in both readers, or ``/info`` raises mid-table on a host that is
      mid-upgrade — the exact host ``/info`` is opened on.
    * ``state`` is subtle: ``stale`` means ``registry.scan`` just DELETED the
      record file, not that the session is idle. A second implementation would
      have got the counters wrong.

    ``scan`` / ``usage`` / ``now`` are injectable seams so the tests can pin the
    degradation matrix without a real machine's sessions under them.

    ``include_stored`` folds in sessions that are NOT running — directories
    under ``config_dir()/sessions/`` with no live record — as ``stored`` rows
    carrying only ``session_id`` / ``conversation_name`` / ``last_activity_s``
    (the process-level fields are meaningless for a dead session and stay at
    their defaults). It is the ``lop sessions --all`` opt-in: the default path
    stays exactly the live-only listing every consumer already reads, and the
    /info panel — which is about the running fleet's cost — never asks for it.
    Stored rows come from the same ``resume.recent_session_rows`` scan the
    ``/resume`` picker uses, so a session is named here by the title the picker
    would show, and a directory the picker hides (subagent scratch) is not a
    session for messaging either. ``stored_limit`` is the CLI's ``--limit``
    verbatim; ``None`` (no flag passed) becomes
    :data:`STORED_SESSIONS_DEFAULT_LIMIT` inside ``_stored_lines``.
    """
    from local_operator.mobile.resources import session_resource_usage
    from local_operator.session.runtime import registry

    scan_fn = scan or registry.scan
    usage_fn = usage or session_resource_usage
    scanned = scan_fn(root)
    stamp = time.time() if now is None else now
    live_pids = [rec.pid for rec, state in scanned if state == "live"]
    measured = usage_fn(live_pids)

    lines: list[SessionLine] = []
    for rec, state in scanned:
        use = measured.get(rec.pid)
        lines.append(
            SessionLine(
                state=state,
                pid=rec.pid,
                kind=rec.kind,
                conversation_name=rec.conversation_name,
                session_id=rec.session_id,
                model_label=rec.model_label,
                cwd=rec.cwd,
                rss_bytes=use.rss_bytes if use else None,
                footprint_bytes=use.footprint_bytes if use else None,
                uptime_s=max(0.0, stamp - rec.started_at),
                heartbeat_age_s=max(0.0, stamp - rec.heartbeat_at),
                # Live state from the record. Defaulted through getattr so a
                # record written by an OLDER runtime (which has no such fields)
                # lists cleanly rather than raising mid-table.
                pending=getattr(rec, "pending", None),
                busy=bool(getattr(rec, "busy", False)),
                detached=bool(getattr(rec, "detached", False)),
                # Which build each runtime is running, for diagnosing skew
                # across a host that replaces its install several times a day.
                # Same getattr defaulting as the live-state fields above.
                version=getattr(rec, "version", "") or "",
                source_ref=getattr(rec, "source_ref", "") or "",
                # NOT coerced to 0 — see ``_reported_count``. The default is
                # ``None`` and stays ``None``: a runtime predating these fields
                # has not told us it has no subagents, and ``or 0`` here would
                # silently turn every older peer into a confident zero in the
                # fleet total.
                subagents_running=_reported_count(getattr(rec, "subagents_running", None)),
                subagents_queued=_reported_count(getattr(rec, "subagents_queued", None)),
                is_self=self_pid is not None and rec.pid == self_pid,
            )
        )

    if include_stored and root is not None:
        lines.extend(_stored_lines(root, {line.session_id for line in lines}, stored_limit))

    # The roll-up counters describe the RUNNING fleet, so they count only the
    # rows the registry published. A ``stored`` row (``include_stored``) names a
    # session that is not running; counting it under ``total``/``live`` would
    # make the /info header report sessions that have no runtime, the exact
    # misreading those counters exist to prevent. ``lines`` keeps them (the
    # listing is the point of the flag); the counters ignore them.
    live_only = [line for line in lines if line.state != "stored"]
    builds = {(line.version, line.source_ref) for line in live_only if line.state == "live"}
    builds.discard(("", ""))

    # The population is every RUNNING PROCESS: ``live`` and ``wedged`` both
    # name a pid that exists, and a runtime that has merely gone quiet for 45 s
    # can still have children burning tokens — dropping it would under-report
    # exactly the fleet somebody opens this screen to understand. It serves its
    # last published counts and the renderer's caveat says they are as of its
    # last heartbeat. A ``stale`` record is excluded outright: the scan above
    # just DELETED its file because the pid is gone, so it is not a runtime.
    live_lines = [line for line in live_only if line.state in ("live", "wedged")]
    reporting = [line for line in live_lines if line.subagents_running is not None]
    fleet_running = sum(line.subagents_running or 0 for line in reporting)
    # Session trajectories and subagent trajectories are DISJOINT by
    # construction, which is what makes adding them legal: a subagent never
    # publishes a SessionRecord of its own (``harness.subagent`` builds a bare
    # Session with no registrant), so nothing is counted both as a runtime's
    # own turn and as somebody's child.
    session_trajectories = sum(1 for line in live_lines if line.busy)
    return SessionsInfo(
        lines=tuple(lines),
        total=len(live_only),
        live=sum(1 for line in live_only if line.state == "live"),
        wedged=sum(1 for line in live_only if line.state == "wedged"),
        stale=sum(1 for line in live_only if line.state == "stale"),
        busy=sum(1 for line in live_only if line.busy),
        pending=sum(1 for line in live_only if line.pending),
        detached=sum(1 for line in live_only if line.detached),
        build_skew=len(builds) > 1,
        # "Nothing measured at all" is a different fact from "this pid could
        # not be measured", and only the first should suppress the column.
        usage_available=not live_pids
        or any(line.rss_bytes is not None or line.footprint_bytes is not None for line in lines),
        subagents_reporting=len(reporting),
        subagents_unreported=len(live_lines) - len(reporting),
        fleet_subagents_running=fleet_running,
        fleet_subagents_queued=sum(line.subagents_queued or 0 for line in reporting),
        fleet_session_trajectories=session_trajectories,
        fleet_trajectories=session_trajectories + fleet_running,
    )


#: How many stored sessions ``lop sessions --all`` lists when no ``--limit`` is
#: given. The store grows without bound on a well-used machine, so the listing
#: caps the rows it shows to the most recent rather than printing a wall; the
#: flag is an opt-in and the cap is named so a consumer can see where it is
#: chosen. Applied inside ``_stored_lines`` — the CLI passes ``None`` for
#: "no --limit given" and relies on this default; a caller that wants a
#: different number passes its own at its own call site, and there is no
#: uncapped spelling (review round 1, MAJOR-2: the constant shipped unwired
#: and ``--all`` listed the entire store).
STORED_SESSIONS_DEFAULT_LIMIT = 50


def _stored_lines(root: Path, live_ids: set[str], limit: int | None) -> list[SessionLine]:
    """One ``SessionLine`` per STORED session: a directory with no live record.

    Read from the same ``resume.recent_session_rows`` scan the ``/resume``
    picker uses, so a session is named here by the title the picker would show
    and a directory the picker hides (subagent scratch) is not offered for
    messaging either. The scan is newest-first on the transcript activity clock
    (``session.retention.session_activity``) — the recency a human recognises
    — so the cap truncates the tail of an ordering that is already useful.

    Only the durable identity fields are filled: ``pid``, RSS, uptime and
    heartbeat age are properties of a RUNNING process and are meaningless for a
    dead session, so they keep the empty defaults the renderer turns into ``—``.
    ``last_activity_s`` carries the transcript mtime in their place.

    Best-effort, never a gate: a store that cannot be read yields no stored rows
    rather than failing a listing whose first job is the LIVE fleet.

    ``limit`` is the CLI's ``--limit`` verbatim when given; ``None`` means no
    flag was passed and becomes :data:`STORED_SESSIONS_DEFAULT_LIMIT` HERE
    rather than in argparse, so the number a consumer reads out of the listing
    is the number this module chose and advertises in ``--limit``'s help. The
    caller validates positivity; a given value is forwarded as-is because
    ``recent_session_rows`` owns slicing semantics.
    """
    from local_operator.resume import recent_session_rows

    resolved = STORED_SESSIONS_DEFAULT_LIMIT if limit is None else limit
    try:
        rows = recent_session_rows(root, resolved)
    except Exception:  # noqa: BLE001 — a listing must not fail on the store
        return []
    lines: list[SessionLine] = []
    for row in rows:
        # Live wins: a session the registry just published for is not "stored",
        # and listing it twice would double-count one conversation in the
        # output the operator reads to decide what to send where.
        if row.id in live_ids:
            continue
        lines.append(
            SessionLine(
                pid=0,
                state="stored",
                session_id=row.id,
                conversation_name=row.name,
                last_activity_s=row.mtime,
            )
        )
    return lines


def session_rows(
    root: Path | None = None,
    *,
    include_stored: bool = False,
    stored_limit: int | None = None,
) -> list[dict[str, Any]]:
    """``lop sessions --json``'s rows, in its established key order.

    The CLI's ``--json`` contract is a published surface, so the key order and
    the key names are pinned here (``tests/unit/info/test_sessions_extraction.py``
    compares against the pre-extraction literal) rather than derived from
    ``dataclasses.asdict``, which would leak ``is_self`` — a field the CLI never
    had — into it.

    ``include_stored`` appends ``stored`` rows for sessions with a directory but
    no live record (the ``--all`` opt-in). The published shape is EXTENDED, not
    broken: the established keys stay first and in order, and the stored-only
    ``last_activity_s`` rides at the END so every existing key position is
    preserved. It is present on every row (``None`` on a live one) because a
    consumer that must branch on key EXISTENCE per row is a worse contract than
    a stable shape with one nullable field. The extraction test asserts exactly
    this order, so the expectation is updated in the same change.
    """
    info = collect_sessions(root, include_stored=include_stored, stored_limit=stored_limit)
    return [
        {
            "state": line.state,
            "pid": line.pid,
            "kind": line.kind,
            "conversation_name": line.conversation_name,
            "session_id": line.session_id,
            "model_label": line.model_label,
            "cwd": line.cwd,
            "rss_bytes": line.rss_bytes,
            "footprint_bytes": line.footprint_bytes,
            "uptime_s": line.uptime_s,
            "heartbeat_age_s": line.heartbeat_age_s,
            "pending": line.pending,
            "busy": line.busy,
            "detached": line.detached,
            "version": line.version,
            "source_ref": line.source_ref,
            # Added deliberately, not incidentally: ``--json`` is a published
            # surface whose key list is pinned against a pre-extraction
            # snapshot, so adding keys breaks that test BY DESIGN and the
            # expectation is updated in the same change. A fleet consumer
            # counting trajectories across a host wants these, and ``null``
            # (not 0) is what an older runtime contributes — the same
            # unreported-vs-zero distinction the screen makes.
            "subagents_running": line.subagents_running,
            "subagents_queued": line.subagents_queued,
            # The stored row's only clock (transcript activity mtime); ``None``
            # on every live/wedged/stale row, which carries ``uptime_s`` and
            # ``heartbeat_age_s`` instead. Last in the dict so the pinned order
            # above is untouched.
            "last_activity_s": line.last_activity_s,
        }
        for line in info.lines
    ]


def build_subagent_tree(
    nodes: Sequence[Any],
    *,
    max_depth: int = MAX_TREE_DEPTH,
) -> tuple[tuple[SubagentLine, ...], int, int]:
    """Flatten a comms roster into ``(rows, max_depth_seen, nodes_below_cap)``.

    Built from ``comms.nodes()`` plus ``parent_job_id``, NOT by recursing job
    managers, because nested launches land in the ROOT session's single records
    map tagged with their true parent's job id — ``nodes()``' own docstring says
    consumers need the complete roster since the root event stream "can only
    ever describe direct children". So one flat read already contains every
    depth.

    The walk carries a ``seen`` set for the same reason ``comms.ancestors()``
    does: a restored snapshot can carry a malformed edge, and a cycle here would
    hang the screen rather than merely misdraw it. Orphans — a node whose parent
    is not in the roster, which the settled-record eviction can produce — are
    walked from the root rather than silently dropped, since a running subagent
    missing from a "what is running" screen is the worst possible error.

    Ordering is running first, then queued, then everything settled: that is the
    question being asked, mirroring ``sort_needs_you_first``'s urgency-first
    principle.
    """
    # Function-local like every other cross-package import here (see the module
    # docstring). ``runtime.types`` is stdlib-only by contract, but the
    # convention is what keeps that true.
    from local_operator.session.runtime.types import RUNNING_SUBAGENT_STATUSES

    children: dict[str | None, list[Any]] = defaultdict(list)
    known = {getattr(node, "job_id", "") for node in nodes}
    for node in nodes:
        parent = getattr(node, "parent_job_id", None)
        # An orphan is re-parented to the root: its parent record was evicted,
        # not its existence.
        children[parent if parent in known else None].append(node)

    def rank(node: Any) -> tuple[int, str]:
        # The SHARED predicate, not a local literal: the header counts
        # ``pausing`` as a running trajectory, so a local tuple that omitted it
        # sorted a counted-as-running child below the settled rows — the tree
        # disagreeing with the tally directly above it, which is the one error
        # this section must never make.
        status = str(getattr(node, "status", "") or "")
        order = 0 if status in RUNNING_SUBAGENT_STATUSES else 1 if status == "queued" else 2
        return order, str(getattr(node, "label", "") or "")

    rows: list[SubagentLine] = []
    seen: set[str] = set()
    deepest = 0
    below_cap = 0

    def walk(parent: str | None, depth: int) -> None:
        nonlocal deepest, below_cap
        for node in sorted(children.get(parent, ()), key=rank):
            job_id = str(getattr(node, "job_id", "") or "")
            if job_id in seen:
                continue
            seen.add(job_id)
            deepest = max(deepest, depth)
            if depth >= max_depth:
                below_cap += 1
            else:
                rows.append(
                    SubagentLine(
                        job_id=job_id,
                        label=str(getattr(node, "label", "") or ""),
                        status=str(getattr(node, "status", "") or ""),
                        depth=depth,
                        agent_role=str(getattr(node, "agent_role", "") or ""),
                        effort=str(getattr(node, "effort", "") or ""),
                        parent_job_id=getattr(node, "parent_job_id", None),
                        session_id=getattr(node, "session_id", None),
                        live=bool(getattr(node, "live", False)),
                    )
                )
            walk(job_id, depth + 1)

    walk(None, 0)

    # A CYCLE reaches nothing from the root: every node's parent is present, so
    # nothing buckets under ``None`` and the walk above iterates an empty list.
    # The ``seen`` set stops the hang but does not restore the nodes, so the
    # header said "2 running" over an empty tree — one screen contradicting
    # itself, and by this function's own docstring the worst error it can make
    # (review round 1, M1). Anything still unreached is emitted at depth 0, the
    # same treatment an orphan gets: a malformed edge costs indentation, never
    # existence.
    for node in sorted(nodes, key=rank):
        job_id = str(getattr(node, "job_id", "") or "")
        if job_id in seen:
            continue
        seen.add(job_id)
        rows.append(
            SubagentLine(
                job_id=job_id,
                label=str(getattr(node, "label", "") or ""),
                status=str(getattr(node, "status", "") or ""),
                depth=0,
                agent_role=str(getattr(node, "agent_role", "") or ""),
                effort=str(getattr(node, "effort", "") or ""),
                parent_job_id=getattr(node, "parent_job_id", None),
                session_id=getattr(node, "session_id", None),
                live=bool(getattr(node, "live", False)),
            )
        )

    return tuple(rows), deepest, below_cap


def _require_root(root: Path) -> Path:
    """``root``, or RAISE when it is the unreadable sentinel.

    Some registries walk a missing directory and return an empty list instead of
    raising, which turns "we could not resolve the config dir" into an
    authoritative-looking ``0``. Forcing the failure here routes those probes
    through :func:`_safe` like every other one, so the field renders as unknown
    and is named in the degraded block (QA round 2, Q8).
    """
    if root == _UNREADABLE_ROOT:
        raise OSError("config dir could not be resolved")
    return root


def collect_agents(
    live: "LiveState",
    errors: list[tuple[str, str]],
    sessions: SessionsInfo | None = None,
) -> AgentsInfo:
    """Profile and team counts (filesystem walks) over the live tree.

    The TREE itself is not read here — it was captured on the event loop by
    :func:`collect_live`, deliberately, so a ``/new`` during this ~900 ms
    snapshot cannot swap the session under an already-painted header.

    ``sessions`` is the already-completed scan, threaded in only to answer
    whether anyone ELSE reported subagent counts. It is optional so this stays
    callable on its own, and its absence means "no fleet knowledge", which is
    the honest reading of not having looked.
    """
    from local_operator.agents import AgentRegistry
    from local_operator.paths import config_dir
    from local_operator.teams import TeamRegistry

    # Guarded like any other probe: `config_dir()` is `Path.home() / ...` and
    # `Path.home()` RAISES when the home directory cannot be resolved, so a bare
    # call here escaped the block and crashed the app (review round 1, B1).
    # `_UNREADABLE_ROOT` rather than `Path(".")` — see `collect_env`.
    root = _safe("agents.config_dir", config_dir, _UNREADABLE_ROOT, errors)
    return AgentsInfo(
        # 12.1 ms and 2.3 ms respectively on this host: filesystem walks, hence
        # the worker thread rather than the paint path.
        profiles=_safe(
            "agents.profiles",
            lambda: len(AgentRegistry(_require_root(root)).list_agents()),
            0,
            errors,
        ),
        # `_require_root` rather than a bare call: `TeamRegistry.list_teams()`
        # does NOT raise on a directory that does not exist, it returns 0. So on
        # an unresolvable home `teams` was a plausible zero with NO degraded
        # entry naming it anywhere — the one field on the screen with no honest
        # marker at all, where `profiles` and `credentials` at least raised and
        # were caught (QA round 2, Q8). A count read out of a config root we
        # could not resolve is not a measurement, whatever the walk returns.
        teams=_safe(
            "agents.teams", lambda: len(TeamRegistry(_require_root(root)).list_teams()), 0, errors
        ),
        tree=live.tree,
        running=live.running,
        queued=live.queued,
        settled=live.settled,
        max_running=live.max_running,
        at_capacity=live.at_capacity,
        max_depth=live.max_depth,
        deeper=live.deeper,
        roster_unread=bool(getattr(live, "roster_unread", False)),
        # Whether at least one OTHER live runtime reported its counts — not
        # whether the feature exists. The note this gates claims knowledge of
        # other windows, so on a single-session host, or one where every peer
        # runs an older build, it must stay False and the screen must keep
        # saying only this session is visible.
        cross_session_known=(
            sessions is not None
            and any(
                # The same population the roll-ups use: a running pid, whether
                # or not it has gone quiet. A ``stale`` record's process is
                # gone, so its counts describe nothing.
                line.state in ("live", "wedged")
                and not line.is_self
                and line.subagents_running is not None
                for line in sessions.lines
            )
        ),
    )


def collect_env(live: "LiveState", errors: list[tuple[str, str]]) -> EnvInfo:
    """The bug-report extras. Names only — never a credential value."""
    from local_operator.browser_bridge import state as bridge_state
    from local_operator.credentials import CredentialManager
    from local_operator.guides.discovery import discover_guides
    from local_operator.paths import config_dir

    # Same guard as `collect_agents`, for the same reason (review round 1, B1).
    # The fallback is `_UNREADABLE_ROOT`, NOT `Path(".")`: `CredentialManager`
    # CREATES its store on construction, so a relative fallback made this
    # read-only diagnostic write a `credentials.env` into whatever directory the
    # user happened to be in — observed for real while testing the B1 guard.
    # A diagnostic that mutates the machine it is describing is the same class
    # of fault as `check_latest()` rewriting the cache, which §2.1 bans.
    root = _safe("env.config_dir", config_dir, _UNREADABLE_ROOT, errors)

    def browser() -> tuple[str, str, bool]:
        # The FILE classification (0.06 ms, cannot hang), not
        # ``backend.bridge_browser_reachable``, which issues a bounded loopback
        # request on the STALE branch and is async. A diagnostic reports what the
        # file says and labels the uncertainty; resolving it is /browser's job.
        kind, current = bridge_state.liveness(root)
        label = {
            bridge_state.Liveness.FRESH: "extension",
            bridge_state.Liveness.STALE: "extension (stale)",
        }.get(kind, "none")
        return (
            label,
            getattr(current, "browser_name", "") or "",
            bool(getattr(current, "paired", False)),
        )

    def mobile() -> tuple[bool, bool, int | None]:
        # NOT ``install.status()``: it costs 37.2 ms because it also does
        # ``_bundle_state()`` and its OWN ``registry.scan()`` — a second scan
        # this snapshot already paid for — and shells ``launchctl`` on some
        # paths. These two targeted calls are 0.83 ms combined.
        from local_operator.mobile import install as mobile_install

        installed = mobile_install.plist_path().exists()
        # The port is only a MEASUREMENT when something is installed to listen
        # on it. Publishing the module default regardless made an unconfigured
        # host report a plausible number for a relay it does not run — the
        # "absent is not a measured value" rule this screen inherits from
        # ``/session`` (review round 1, N2).
        port = mobile_install.DEFAULT_PORT if installed else None
        healthy = bool(installed and port and mobile_install.health(port))
        return installed, healthy, port

    backend, browser_name, paired = _safe("env.browser", browser, ("none", "", False), errors)
    installed, healthy, port = _safe("env.mobile", mobile, (False, False, None), errors)
    return EnvInfo(
        mcp_configured=live.mcp_configured,
        mcp_connected=live.mcp_connected,
        mcp_failed=live.mcp_failed,
        mcp_settling=live.mcp_settling,
        mcp_failures=live.mcp_failures,
        approval_mode=live.approval_mode,
        theme=live.theme,
        terminal_size=live.terminal_size,
        term=os.environ.get("TERM", ""),
        colorterm=os.environ.get("COLORTERM", ""),
        multiplexer=_multiplexer(),
        is_tty=_safe("env.tty", sys.stdout.isatty, False, errors),
        browser_backend=backend,
        browser_name=browser_name,
        browser_paired=paired,
        mobile_installed=installed,
        mobile_healthy=healthy,
        mobile_port=port,
        # KEY NAMES ONLY. ``get_credentials`` returns SecretStr values and this
        # screen is pasted into issues; a name answers the diagnostic question
        # ("is it even set?") and a value answers nothing this screen asks.
        credential_keys=_safe(
            "env.credentials",
            lambda: tuple(CredentialManager(root).list_credential_keys()),
            (),
            errors,
        ),
        guides=_safe("env.guides", lambda: len(discover_guides()), 0, errors),
        skills=live.skills,
    )


#: Stand-in config root for when `config_dir()` itself cannot be resolved.
#: Deliberately a path that cannot exist and cannot be created, so a probe whose
#: constructor would otherwise MATERIALISE a store (``CredentialManager`` writes
#: a ``credentials.env``) fails into `_safe` and is reported as degraded,
#: instead of silently writing into the process's current directory. `/info`
#: reads; it must never leave anything behind on the host it is describing.
_UNREADABLE_ROOT = Path("/nonexistent/local-operator-info-unreadable-config-root")


def _multiplexer() -> str:
    """Which multiplexer this terminal is inside, by marker PRESENCE only.

    A marker's value carries a socket path or a workspace id; neither is
    diagnostic and both are identifying, so only presence is consulted and the
    value is never read, never stored and never rendered.

    Each read is spelled out rather than looped over
    :data:`_MULTIPLEXER_MARKERS`, because the ambient-environment audit resolves
    reads through the AST and cannot follow a loop variable — written as a loop,
    these reads were invisible to the guard that exists to account for exactly
    them (QA round 1, Q4). The tuple remains the ORDER of preference (most
    specific first) and this function is checked against it by
    ``test_every_multiplexer_marker_is_probed``, so the two cannot drift.
    """
    if os.environ.get(_ENV_CMUX_SOCKET_PATH) is not None:
        return "cmux"
    if os.environ.get(_ENV_CMUX_WORKSPACE_ID) is not None:
        return "cmux"
    if os.environ.get(_ENV_TMUX) is not None:
        return "tmux"
    if os.environ.get(_ENV_STY) is not None:
        return "screen"
    if os.environ.get(_ENV_ZELLIJ) is not None:
        return "zellij"
    if os.environ.get(_ENV_WEZTERM_PANE) is not None:
        return "wezterm"
    return ""


class LiveState:
    """In-memory state captured ON the event loop, before the worker yields.

    Every field here is a scalar or an immutable tuple read from a live object
    in one synchronous pass — the same rule, and for the same reason, as
    ``SessionDiagnostics.capture``: "never hold a mutable session across a disk
    read". The subagent tree in particular MUST be captured here rather than in
    the worker. It is an in-memory dict walk (free), and a ``/new`` or
    ``/resume`` arriving during the ~900 ms session probe would otherwise put a
    brand-new tree under a header describing the old session.
    """

    __slots__ = (
        "session_id",
        "conversation_name",
        "model_label",
        "effective_model",
        "kind",
        "uptime_s",
        "tree",
        "running",
        "queued",
        "settled",
        "max_running",
        "at_capacity",
        "max_depth",
        "deeper",
        "roster_unread",
        "errors",
        "mcp_configured",
        "mcp_connected",
        "mcp_failed",
        "mcp_settling",
        "mcp_failures",
        "approval_mode",
        "theme",
        "terminal_size",
        "skills",
    )

    def __init__(self, **values: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, values.get(name, _LIVE_DEFAULTS[name]))


#: Defaults chosen so a ``LiveState()`` with nothing supplied renders a screen
#: of ``—`` and crashes nothing — the same contract ``model.py`` states.
_LIVE_DEFAULTS: dict[str, Any] = {
    "session_id": "",
    "conversation_name": "",
    "model_label": "",
    "effective_model": "",
    "kind": "tui",
    "uptime_s": None,
    "tree": (),
    "running": 0,
    "queued": 0,
    "settled": 0,
    "max_running": None,
    "at_capacity": False,
    "max_depth": 0,
    "deeper": 0,
    #: The roster could not be READ, which is not the same fact as an empty
    #: roster. Held on the live capture because the tree is drawn from it on
    #: the first frame, before the worker's snapshot exists.
    "roster_unread": False,
    #: Probe failures from the live pass, forwarded into the snapshot's
    #: ``degraded`` block. They used to be collected and DROPPED — the list was
    #: local to ``collect_live`` and had nowhere to go — so a live probe that
    #: failed left the screen with a default and no disclosure anywhere.
    "errors": (),
    "mcp_configured": 0,
    "mcp_connected": 0,
    "mcp_failed": 0,
    "mcp_settling": False,
    "mcp_failures": (),
    "approval_mode": "",
    "theme": "",
    "terminal_size": None,
    "skills": 0,
}


def collect_live(
    session: Any,
    *,
    approve_all: bool = False,
    theme: str = "",
    size: tuple[int, int] | None = None,
    skills: int = 0,
) -> LiveState:
    """Snapshot the live objects synchronously. Safe on the paint path.

    Everything read here is an in-memory attribute or dict walk. Nothing touches
    the filesystem, a socket or a subprocess — that is what makes it legitimate
    on the event loop, and what the caller relies on when it pushes the screen
    before starting the worker.
    """
    # Function-local like every other cross-package import here (see the
    # module docstring). ``runtime.types`` is stdlib-only by contract, but the
    # convention is what keeps that true.
    from local_operator.session.runtime.types import RUNNING_SUBAGENT_STATUSES

    errors: list[tuple[str, str]] = []
    tree: tuple[SubagentLine, ...] = ()
    deepest = 0
    deeper = 0
    running = queued = settled = 0
    max_running: int | None = None
    at_capacity = False

    # ``_attr`` and not a bare ``getattr``: every one of these is a PROPERTY on
    # the real Session, and a property on an unhealthy session can raise. A bare
    # getattr would let that propagate out of a function whose entire contract
    # is "safe on the paint path", turning a wedged session into a crash on the
    # one screen that exists to describe it.
    comms = _attr(session, "subagent_comms", None) if session is not None else None
    roster_unread = False
    if session is not None and comms is None:
        # A session object that cannot answer for its own roster. This was
        # silent: the branch below was skipped, the counts stayed 0 and NOTHING
        # was recorded, so the screen stated "no subagents have been launched"
        # about a session it had never asked. That is the one failure mode
        # ``_safe`` does not catch, because nothing raised — the default was
        # simply returned and believed. Named here so it renders as UNKNOWN and
        # appears in the degraded block, like every probe that fails loudly.
        roster_unread = True
        errors.append(("live.subagents", "session exposes no subagent_comms"))
    elif comms is not None:
        # ``lambda: comms.nodes()`` and not ``comms.nodes``: the bare attribute
        # is resolved as an ARGUMENT, before ``_safe`` enters its try, so a
        # comms object lacking the method raised straight out of a function
        # whose whole contract is "safe on the paint path".
        nodes = _safe("live.subagents", lambda: list(comms.nodes()), None, errors)
        if nodes is None:
            # The read was attempted and failed. ``_safe`` already named it in
            # ``errors``; the flag is what stops the renderer printing a
            # confident zero over a roster nobody managed to read.
            roster_unread = True
            nodes = []
        tree, deepest, deeper = build_subagent_tree(nodes)
        for node in nodes:
            status = str(getattr(node, "status", "") or "")
            # The shared predicate, not a local literal: the runtime publishes
            # ``subagents_running`` from the same set, and a divergence would
            # put a fleet total on the header that disagrees with the tree
            # drawn directly beneath it.
            if status in RUNNING_SUBAGENT_STATUSES:
                running += 1
            elif status == "queued":
                queued += 1
            else:
                settled += 1

    jobs = _attr(session, "jobs", None) if session is not None else None
    if jobs is not None:
        # A follower's ``jobs`` is ``SnapshotJobs`` — a minimal facade that only
        # knows the roster, NOT the capacity fields the owner's
        # ``AsyncJobManager`` carries. Probing them unconditionally on a follower
        # surfaced two "Could not read: AttributeError" rows on a HEALTHY
        # screen, which is exactly the false alarm this collector exists to
        # prevent. Leaving both at their defaults instead keeps the panel
        # honest: ``max_running=None`` hides the "Subagent capacity" row rather
        # than asserting the built-in cap the owner may have overridden, and
        # ``at_capacity=False`` simply declines to warn.
        if hasattr(jobs, "max_running"):
            max_running = _safe("live.max_running", lambda: int(jobs.max_running), None, errors)
            at_capacity = _safe("live.at_capacity", lambda: bool(jobs.at_capacity()), False, errors)

    startup = _attr(session, "mcp_startup", None) if session is not None else None
    failures: dict[str, str] = dict(_attr(startup, "failures", {}) or {})
    return LiveState(
        session_id=_attr(session, "session_id", ""),
        conversation_name=_attr(session, "conversation_name", ""),
        model_label=_attr(session, "model_label", ""),
        effective_model=_attr(session, "effective_model_label", ""),
        uptime_s=None,
        tree=tree,
        running=running,
        queued=queued,
        settled=settled,
        max_running=max_running,
        at_capacity=at_capacity,
        max_depth=deepest,
        deeper=deeper,
        roster_unread=roster_unread,
        errors=tuple(errors),
        mcp_configured=len(getattr(startup, "configured", ()) or ()),
        mcp_connected=len(getattr(startup, "connected", ()) or ()),
        mcp_failed=len(failures),
        mcp_settling=bool(getattr(startup, "settling", False)),
        # Server NAMES and the failure MESSAGE only, truncated: a message can
        # quote a command line, and the ones that matter ("command not found:
        # gh") are short.
        mcp_failures=tuple((name, str(text)[:120]) for name, text in sorted(failures.items())),
        approval_mode="auto" if approve_all else "ask",
        theme=theme,
        terminal_size=tuple(size) if size else None,  # type: ignore[arg-type]
        skills=skills,
    )


def _attr(obj: Any, name: str, default: T) -> T:
    """Read one public attribute, tolerating a facade that lacks it or a property
    that RAISES.

    The raising case is the one that matters and the one a bare ``getattr``
    misses: ``session_id``, ``model_label`` and ``subagent_comms`` are all
    properties on the real ``Session``, so a session that is itself unhealthy —
    exactly the session someone opens ``/info`` about — would otherwise take the
    screen down with it.
    """
    try:
        value = getattr(obj, name, default)
    except Exception:  # noqa: BLE001 — an unhealthy session still gets a screen
        return default
    return default if value is None else value


def collect_snapshot(live: LiveState, *, root: Path | None = None) -> InfoSnapshot:
    """Every blocking probe, on a worker thread. Never call this on the loop.

    Blocking is the point: ``session_resource_usage`` alone measured 879.5 ms
    for 12 pids (it shells ``top -l1`` for the whole system on macOS, a trade
    documented in ``mobile/resources.py``). Each block is guarded independently
    so one wedged filesystem costs its own section and nothing else.
    """
    # Seeded with the LIVE pass's failures rather than starting empty: those
    # probes ran on the event loop where this function cannot reach them, and
    # dropping them meant a failure with a named reason vanished before the
    # degraded block that exists to print it.
    errors: list[tuple[str, str]] = list(getattr(live, "errors", ()) or ())
    self_pid = os.getpid()
    sessions = _safe(
        "sessions",
        lambda: collect_sessions(root, self_pid=self_pid),
        SessionsInfo(available=False),
        errors,
    )
    self_line = next((line for line in sessions.lines if line.is_self), None)
    # EVERY block call is wrapped, not just the probes inside them. The
    # individual `_safe`s within each collector guard only what is inside their
    # lambdas; a block function's own prologue — its function-local imports and
    # its `config_dir()` call — was bare, so an unresolvable home
    # (`Path.home()` raises `RuntimeError`) or a broken submodule propagated out
    # of here. That is worse than an empty screen: `run_worker` defaults to
    # `exit_on_error=True`, so the exception took the whole APP down, on exactly
    # the broken host `/info` exists to describe (review round 1, B1). The
    # fallbacks are all-defaults instances, which `model.py` guarantees render.
    return InfoSnapshot(
        install=_safe("install", lambda: collect_install(errors), InstallInfo(), errors),
        process=_safe(
            "process",
            lambda: collect_process(live, self_line=self_line, errors=errors, root=root),
            ProcessInfo(),
            errors,
        ),
        sessions=sessions,
        agents=_safe(
            "agents", lambda: collect_agents(live, errors, sessions), AgentsInfo(), errors
        ),
        env=_safe("env", lambda: collect_env(live, errors), EnvInfo(), errors),
        degraded=tuple(errors),
        captured_at=time.time(),
    )


def collect_info(
    session: Any = None,
    *,
    approve_all: bool = False,
    theme: str = "",
    size: tuple[int, int] | None = None,
    root: Path | None = None,
) -> InfoSnapshot:
    """Both phases at once — for the CLI and for tests, never for the TUI.

    The TUI must keep the two apart (``collect_live`` on the loop,
    ``collect_snapshot`` in a worker) so the screen paints before the 900 ms
    probe. A non-interactive caller has no frame to protect and can pay for both
    in one go.
    """
    live = collect_live(session, approve_all=approve_all, theme=theme, size=size)
    return collect_snapshot(live, root=root)


def degraded_names(errors: Iterable[tuple[str, str]]) -> tuple[str, ...]:
    """Just the field names that failed, for a compact footnote."""
    return tuple(name for name, _ in errors)

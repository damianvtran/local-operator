"""The ``/info`` probes: no network, honest staleness, independent degradation.

The highest-value assertion in this file is
:func:`test_collect_never_touches_the_network`. ``/info``'s whole design turns on
it, and a comment cannot enforce it: ``update.check_latest()`` is the
obvious-looking call, its name reads as harmless, and its TTL miss is a live
5 s HTTP request that also rewrites the cache. It is pinned STRUCTURALLY (a fake
that raises) rather than with a wall-clock bound, per AGENTS.md §"Prefer a
structural invariant to a numeric one" — a timing bound on a ``top``-shelling
probe would flake on CI within a day.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from local_operator import update
from local_operator.info import collect as collect_mod
from local_operator.info.collect import (
    LiveState,
    build_subagent_tree,
    collect_agents,
    collect_env,
    collect_install,
    collect_live,
    collect_sessions,
    collect_snapshot,
)
from local_operator.info.model import InfoSnapshot, SessionsInfo
from local_operator.session.protocol import RuntimeLocality

#: Frozen stamp for fixtures whose durations must be constants.
NOW_FIXTURE = 1_788_602_400.0


@dataclass
class _Record:
    """A ``SessionRecord``-shaped stand-in. Deliberately NOT the real class for
    the mid-upgrade test below, whose whole point is a record missing fields."""

    pid: int
    kind: str = "tui"
    session_id: str = "sess"
    conversation_name: str = "Conversation"
    cwd: str = "/tmp/x"
    model_label: str = "anthropic/claude-opus-5"
    started_at: float = 0.0
    heartbeat_at: float = 0.0
    pending: str | None = None
    busy: bool = False
    detached: bool = False
    version: str = "0.51.6"
    source_ref: str = "abc1234"
    subagents_running: int | None = None
    subagents_queued: int | None = None


@dataclass
class _OldRecord:
    """What an OLDER runtime wrote: no version/source_ref/pending/busy/detached.

    This is the exact shape the ``getattr`` defaulting exists for, and the host
    it appears on — one mid-upgrade — is precisely the host ``/info`` is opened
    on.
    """

    pid: int
    kind: str
    session_id: str
    conversation_name: str
    cwd: str
    model_label: str
    started_at: float
    heartbeat_at: float


@dataclass
class _Usage:
    rss_bytes: int | None = None
    footprint_bytes: int | None = None


@dataclass
class _Node:
    job_id: str
    label: str
    parent_job_id: str | None = None
    status: str = "running"
    agent_role: str = ""
    effort: str = ""
    session_id: str | None = None
    live: bool = True


# -- the ban on the network ---------------------------------------------------


def test_collect_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """``collect_install`` completes with every fetch path armed to explode.

    Both doors are nailed shut: the module's own fetch helper and ``httpx.get``
    beneath it. If a future edit reaches for ``check_latest()`` — on any branch,
    including ``force=False`` — this fails loudly instead of shipping a screen
    that hangs for five seconds on the broken network it exists to explain.
    """

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("/info must never fetch: it is opened when the network is broken")

    monkeypatch.setattr(update, "_fetch_pypi_version", _explode)
    monkeypatch.setattr(update, "check_latest", _explode)
    import httpx

    monkeypatch.setattr(httpx, "get", _explode)

    errors: list[tuple[str, str]] = []
    install = collect_install(errors)

    assert install.version  # the field a bug report most needs survived
    assert not errors
    # Either something was already cached, or nothing was. Both are fine; a
    # NETWORK CALL is not.
    assert install.latest_known is None or isinstance(install.latest_known, str)


def test_cold_cache_reports_unknown_not_up_to_date(tmp_path: Path) -> None:
    """``(None, None)``, which the screen must render as "never checked".

    Collapsing this into "up to date" tells a user on a broken network that they
    are on the newest release, which is the failure mode ``cached_latest`` was
    added to prevent.
    """
    assert update.cached_latest(tmp_path) == (None, None)


def test_fresh_cache_is_returned_with_its_age(tmp_path: Path) -> None:
    path = tmp_path / "pypi-local-operator.json"
    path.write_text(json.dumps({"payload": {"version": "9.9.9"}, "fetched_at": time.time() - 120}))
    version, age = update.cached_latest(tmp_path)
    assert version == "9.9.9"
    assert age is not None and 100 < age < 200


def test_stale_cache_is_returned_and_its_age_exceeds_the_ttl(tmp_path: Path) -> None:
    """A 7-hour-old document against a 6-hour TTL: still returned, marked old.

    The value is still the best answer available; only its FRESHNESS changed,
    and the renderer appends ``(stale)`` from this age rather than discarding a
    usable version number.
    """
    path = tmp_path / "pypi-local-operator.json"
    fetched = time.time() - (7 * 60 * 60)
    path.write_text(json.dumps({"payload": {"version": "9.9.9"}, "fetched_at": fetched}))
    version, age = update.cached_latest(tmp_path)
    assert version == "9.9.9"
    assert age is not None and age > update.TTL_S


def test_corrupt_cache_is_unknown_rather_than_an_exception(tmp_path: Path) -> None:
    (tmp_path / "pypi-local-operator.json").write_text("{not json")
    assert update.cached_latest(tmp_path) == (None, None)


def test_future_dated_cache_keeps_the_version_and_drops_the_age(tmp_path: Path) -> None:
    """A clock skew makes the age meaningless, not the version."""
    path = tmp_path / "pypi-local-operator.json"
    path.write_text(json.dumps({"payload": {"version": "9.9.9"}, "fetched_at": time.time() + 9999}))
    assert update.cached_latest(tmp_path) == ("9.9.9", None)


# -- sessions -----------------------------------------------------------------


def _scan(records: list[tuple[Any, str]]) -> Any:
    return lambda root=None: records


def _usage(mapping: dict[int, _Usage]) -> Any:
    return lambda pids, **kwargs: {pid: mapping.get(pid, _Usage()) for pid in pids}


def test_counters_agree_with_the_lines() -> None:
    """``live + wedged + stale == total``, and each counter is its own filter.

    Cheap, and it catches the copy-paste drift that a hand-maintained set of
    seven roll-ups invites.
    """
    records = [
        (_Record(pid=1, busy=True), "live"),
        (_Record(pid=2, pending="approval"), "live"),
        (_Record(pid=3, detached=True), "wedged"),
        (_Record(pid=4), "stale"),
    ]
    info = collect_sessions(scan=_scan(records), usage=_usage({}), now=100.0)
    assert info.total == 4
    assert info.live + info.wedged + info.stale == info.total
    assert (info.live, info.wedged, info.stale) == (2, 1, 1)
    assert info.busy == sum(1 for line in info.lines if line.busy) == 1
    assert info.pending == 1
    assert info.detached == 1


def test_is_self_marks_exactly_the_calling_process() -> None:
    records = [(_Record(pid=11), "live"), (_Record(pid=22), "live")]
    info = collect_sessions(scan=_scan(records), usage=_usage({}), self_pid=22, now=0.0)
    assert [line.is_self for line in info.lines] == [False, True]


def test_a_record_missing_the_newer_fields_does_not_raise() -> None:
    """The mid-upgrade host: an old record lists cleanly with empty defaults."""
    old = _OldRecord(
        pid=7,
        kind="exec",
        session_id="old",
        conversation_name="Older runtime",
        cwd="/tmp/old",
        model_label="openai/gpt-5.2",
        started_at=0.0,
        heartbeat_at=0.0,
    )
    info = collect_sessions(scan=_scan([(old, "live")]), usage=_usage({}), now=60.0)
    line = info.lines[0]
    assert line.version == "" and line.source_ref == ""
    assert line.pending is None and line.busy is False and line.detached is False
    assert line.uptime_s == 60.0


def test_build_skew_is_only_flagged_across_live_sessions() -> None:
    same = [
        (_Record(pid=1, version="0.51.6", source_ref="aaa"), "live"),
        (_Record(pid=2, version="0.51.6", source_ref="aaa"), "live"),
        # A STALE record on an older build is not skew: it is not running.
        (_Record(pid=3, version="0.40.0", source_ref="zzz"), "stale"),
    ]
    assert collect_sessions(scan=_scan(same), usage=_usage({}), now=0.0).build_skew is False

    mixed = [
        (_Record(pid=1, version="0.51.6", source_ref="aaa"), "live"),
        (_Record(pid=2, version="0.51.5", source_ref="bbb"), "live"),
    ]
    assert collect_sessions(scan=_scan(mixed), usage=_usage({}), now=0.0).build_skew is True


def test_usage_available_is_false_only_when_nothing_measured() -> None:
    records = [(_Record(pid=1), "live"), (_Record(pid=2), "live")]
    none_measured = collect_sessions(scan=_scan(records), usage=_usage({}), now=0.0)
    assert none_measured.usage_available is False

    partly = collect_sessions(
        scan=_scan(records), usage=_usage({1: _Usage(rss_bytes=1000)}), now=0.0
    )
    assert partly.usage_available is True


def test_no_live_sessions_is_not_an_unmeasurable_host() -> None:
    """Nothing to measure is not the same as a host that cannot measure."""
    info = collect_sessions(scan=_scan([(_Record(pid=9), "stale")]), usage=_usage({}), now=0.0)
    assert info.usage_available is True


# -- the subagent tree --------------------------------------------------------


def test_tree_is_depth_first_and_counts_its_depth() -> None:
    """root → A → A1 → A1a, built from ``parent_job_id`` alone."""
    nodes = [
        _Node("root", "reviewer"),
        _Node("a", "scout", parent_job_id="root"),
        _Node("a1", "coder", parent_job_id="a"),
    ]
    tree, deepest, deeper = build_subagent_tree(nodes)
    assert [(n.job_id, n.depth) for n in tree] == [("root", 0), ("a", 1), ("a1", 2)]
    assert deepest == 2 and deeper == 0


def test_tree_caps_its_depth_and_counts_the_remainder() -> None:
    """Past the cap the COUNT is the fact that matters, not the indentation."""
    nodes = [
        _Node("n0", "a"),
        _Node("n1", "b", parent_job_id="n0"),
        _Node("n2", "c", parent_job_id="n1"),
        _Node("n3", "d", parent_job_id="n2"),
        _Node("n4", "e", parent_job_id="n3"),
    ]
    tree, deepest, deeper = build_subagent_tree(nodes, max_depth=3)
    assert max(node.depth for node in tree) == 2
    assert deepest == 4
    assert deeper == 2


def test_tree_walk_is_cycle_safe() -> None:
    """A malformed restored snapshot must not hang the screen.

    Mirrors the ``seen`` set ``comms.ancestors()`` carries for exactly this.
    """
    nodes = [_Node("x", "X", parent_job_id="y"), _Node("y", "Y", parent_job_id="x")]
    tree, _deepest, _deeper = build_subagent_tree(nodes)
    # `<= 2` PASSED VACUOUSLY AT 0, which is what it did before the fix: in a
    # cycle every node's parent is present, so nothing buckets under `None`, the
    # walk iterates an empty list and the tree comes back EMPTY while the header
    # still says "2 running" — one screen contradicting itself, and by
    # `build_subagent_tree`'s own docstring the worst error it can make (review
    # round 1, M1). Assert the exact count: a malformed edge must cost
    # indentation, never existence.
    assert len(tree) == 2
    assert {node.job_id for node in tree} == {"x", "y"}
    assert len({node.job_id for node in tree}) == len(tree)


def test_an_orphan_is_walked_from_the_root_not_dropped() -> None:
    """Its parent record was evicted, not its existence.

    A RUNNING subagent missing from a "what is running" screen is the worst
    error this screen can make.
    """
    nodes = [_Node("child", "scout", parent_job_id="evicted-parent")]
    tree, _deepest, _deeper = build_subagent_tree(nodes)
    assert [node.job_id for node in tree] == ["child"]
    assert tree[0].depth == 0


def test_running_sorts_before_queued_before_settled() -> None:
    nodes = [
        _Node("c", "done", status="completed"),
        _Node("b", "waiting", status="queued"),
        _Node("a", "working", status="running"),
    ]
    tree, _deepest, _deeper = build_subagent_tree(nodes)
    assert [node.status for node in tree] == ["running", "queued", "completed"]


# -- independent degradation --------------------------------------------------


class _Boom:
    """A session whose every read raises. A diagnostic must survive one."""

    @property
    def subagent_comms(self) -> Any:
        raise RuntimeError("comms is wedged")

    @property
    def session_id(self) -> str:
        raise RuntimeError("session is wedged")


def test_live_capture_survives_a_wedged_session() -> None:
    live = collect_live(_Boom(), theme="dawn", size=(100, 30))
    assert live.tree == ()
    assert live.theme == "dawn"
    assert live.terminal_size == (100, 30)


def test_live_capture_of_none_is_a_full_default_state() -> None:
    live = collect_live(None)
    assert live.running == 0 and live.tree == () and live.max_running is None


@pytest.mark.parametrize("failing", ["sessions", "agents", "env", "install"])
def test_each_block_degrades_independently(failing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make exactly ONE collector raise; every other block must still populate.

    This is the property the whole ``_safe`` design exists for: one unreadable
    probe costs exactly its own field, never the version number a bug report
    most needs.
    """

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{failing} is broken")

    if failing == "sessions":
        monkeypatch.setattr(collect_mod, "collect_sessions", _explode)
    elif failing == "agents":
        import local_operator.agents as agents_mod

        monkeypatch.setattr(agents_mod, "AgentRegistry", _explode)
    elif failing == "env":
        from local_operator.browser_bridge import state as bridge_state

        monkeypatch.setattr(bridge_state, "liveness", _explode)
    else:
        monkeypatch.setattr(update, "installed_version", _explode)

    snapshot = collect_snapshot(collect_live(None, theme="dusk"))

    assert snapshot.degraded, "the failure must be NAMED, not swallowed"
    names = {name for name, _ in snapshot.degraded}
    assert any(failing.split(".")[0] in name for name in names)

    # And nothing propagated: the other blocks are still real objects.
    assert isinstance(snapshot.sessions, SessionsInfo)
    assert snapshot.env.theme == "dusk"
    if failing != "install":
        assert snapshot.install.version


def test_a_broken_scan_replaces_the_section_rather_than_the_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("registry is on a wedged mount")

    monkeypatch.setattr(collect_mod, "collect_sessions", _explode)
    snapshot = collect_snapshot(collect_live(None))
    assert snapshot.sessions.available is False
    assert snapshot.install.version  # unaffected


def test_degraded_reasons_are_named_and_bounded() -> None:
    errors: list[tuple[str, str]] = []
    collect_mod._safe("probe.name", lambda: 1 / 0, "fallback", errors)
    assert errors and errors[0][0] == "probe.name"
    assert "ZeroDivisionError" in errors[0][1]
    assert len(errors[0][1]) <= 120


def test_environment_reports_marker_names_never_marker_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multiplexer marker's VALUE carries a socket path or a workspace id."""
    monkeypatch.setenv("CMUX_SOCKET_PATH", "/private/tmp/secret-socket-path")
    env = collect_env(collect_live(None), [])
    assert env.multiplexer == "cmux"
    assert "secret-socket-path" not in repr(env)


# -- the resolved import path -------------------------------------------------


def test_import_path_is_the_package_directory_actually_resolved() -> None:
    """The field that answers "which code am I running" without inference."""
    import local_operator

    errors: list[tuple[str, str]] = []
    install = collect_install(errors)
    assert install.import_path == str(Path(local_operator.__file__).resolve().parent)
    # In THIS worktree the venv is installed editable from this tree, so the
    # package legitimately sits outside the venv prefix. Asserting the flag's
    # value here would pin the developer's environment rather than the logic;
    # the containment rule itself is pinned below against synthetic paths.
    assert isinstance(install.import_path_foreign, bool)


def test_a_package_under_the_prefix_is_not_foreign(tmp_path: Path) -> None:
    """Containment, not equality: a healthy install nests several levels deep."""
    prefix = tmp_path / "venv"
    package = prefix / "lib" / "python3.12" / "site-packages" / "local_operator"
    package.mkdir(parents=True)
    assert collect_mod._import_path_is_foreign(str(package), str(prefix)) is False


def test_a_checkout_shadowing_the_install_is_foreign(tmp_path: Path) -> None:
    """The ``launch.py:287`` trap: spawned with ``-m`` and no ``cwd=``, so the
    session's own directory wins ``sys.path[0]`` over site-packages."""
    prefix = tmp_path / "uv" / "tools" / "local-operator"
    checkout = tmp_path / "repos" / "local-operator" / "local_operator"
    prefix.mkdir(parents=True)
    checkout.mkdir(parents=True)
    assert collect_mod._import_path_is_foreign(str(checkout), str(prefix)) is True


def test_an_unreadable_comparison_never_invents_a_warning() -> None:
    """A failed comparison must not raise a false alarm on the one screen whose
    job is to be believed."""
    assert collect_mod._import_path_is_foreign("", "/opt/x") is False
    assert collect_mod._import_path_is_foreign("/opt/x", "") is False


def test_collect_sessions_against_a_record_captured_from_a_real_scan() -> None:
    """The fixture is a REAL record's field set, not one imagined from the type.

    Written this way deliberately: the ``/session`` suite shipped a bug green
    because its fixtures asserted an outcome shape the recorder cannot actually
    produce. So this dict is the exact key set ``SessionRecord.to_json()``
    returned from a live scan on this machine — including ``kind='daemon'``,
    which is what every real record carries (``runtime/process.py`` spawns with
    it) and NOT the ``tui`` a reading of the ``Literal["tui", "exec", "daemon"]``
    would suggest.
    """
    from local_operator.session.runtime.types import SessionRecord

    captured = {
        "pid": 62950,
        "kind": "daemon",
        "session_id": "9c1f2ab40e77",
        "conversation_name": "Fix error starting new session",
        "cwd": "/private/tmp/lop-example",
        "model_label": "anthropic/claude-opus-5",
        "control_port": 62950,
        "control_key": "a" * 64,
        "protocol": 5,
        "started_at": NOW_FIXTURE - 3600.0,
        "heartbeat_at": NOW_FIXTURE - 4.0,
        "capabilities": [],
        "busy": True,
        "detached": False,
        "pending": None,
        "version": "0.51.6",
        "source_ref": "4311eb653aa9",
    }
    record = SessionRecord.from_json(captured)
    info = collect_sessions(
        scan=lambda root=None: [(record, "live")],
        usage=lambda pids, **kwargs: {pid: _Usage(rss_bytes=609 * (1 << 20)) for pid in pids},
        now=NOW_FIXTURE,
    )
    line = info.lines[0]
    assert line.kind == "daemon"
    assert line.busy is True and line.pending is None
    assert line.uptime_s == 3600.0 and line.heartbeat_age_s == 4.0
    assert line.version == "0.51.6"
    # And the key present in the REAL record does not survive into the line.
    assert not hasattr(line, "control_key")
    assert "a" * 16 not in repr(line)


def test_every_multiplexer_marker_is_probed() -> None:
    """`_multiplexer` spells its reads out; the tuple stays the source of truth.

    The reads are unrolled so the ambient-environment audit can resolve them
    through the AST (QA round 1, Q4), which creates a second place the marker
    list effectively lives. This pins them together: every marker in the tuple
    must actually be probed, and must map to the label the tuple gives it.
    """
    import os

    for variable, expected in collect_mod._MULTIPLEXER_MARKERS:
        saved = {name: os.environ.pop(name, None) for name, _ in collect_mod._MULTIPLEXER_MARKERS}
        try:
            os.environ[variable] = "x"
            assert collect_mod._multiplexer() == expected, variable
        finally:
            os.environ.pop(variable, None)
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value


def test_a_multiplexer_marker_value_is_never_read() -> None:
    """Presence only: a socket path or workspace id must not reach the snapshot."""
    import os

    saved = {name: os.environ.pop(name, None) for name, _ in collect_mod._MULTIPLEXER_MARKERS}
    try:
        os.environ["CMUX_SOCKET_PATH"] = "/tmp/SENTINEL-SOCKET-VALUE/sock"
        errors: list[tuple[str, str]] = []
        env = collect_env(LiveState(), errors)
        assert env.multiplexer == "cmux"
        assert "SENTINEL-SOCKET-VALUE" not in repr(env)
    finally:
        os.environ.pop("CMUX_SOCKET_PATH", None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def test_a_failure_in_a_block_prologue_degrades_instead_of_escaping() -> None:
    """B1: the prologue was outside every guard, and the escape KILLED the app.

    `test_each_block_degrades_independently` makes the innermost probe raise,
    which lands inside an existing `_safe`. Nothing exercised a failure in a
    block function's own prologue — its function-local imports and its
    `config_dir()` call — and `config_dir()` is `Path.home() / ...`, which
    raises `RuntimeError` when the home cannot be resolved. Escaping
    `collect_snapshot` meant escaping the worker, and `run_worker` defaults to
    `exit_on_error=True`, so the diagnostic screen took the whole application
    down on exactly the broken host it exists to describe (review round 1, B1).
    """
    import local_operator.paths as paths_mod

    def boom(*args: object, **kwargs: object) -> Path:
        raise OSError("home unreadable")

    original = paths_mod.config_dir
    paths_mod.config_dir = boom  # type: ignore[assignment]
    try:
        snapshot = collect_snapshot(LiveState())
    finally:
        paths_mod.config_dir = original  # type: ignore[assignment]

    # It RETURNED rather than raising, and said what it could not read.
    assert isinstance(snapshot, InfoSnapshot)
    assert snapshot.degraded, "a prologue failure must be reported, not swallowed silently"
    assert any("home unreadable" in reason for _name, reason in snapshot.degraded)


def test_a_broken_submodule_degrades_instead_of_escaping() -> None:
    """The other prologue hazard: a function-local import that cannot resolve.

    This is the state `/info` is opened in often enough to matter — a partially
    broken install — so it must render, not crash.
    """
    import sys

    sentinel = object()
    saved = sys.modules.get("local_operator.credentials", sentinel)
    sys.modules["local_operator.credentials"] = None  # type: ignore[assignment]
    try:
        snapshot = collect_snapshot(LiveState())
    finally:
        if saved is sentinel:
            sys.modules.pop("local_operator.credentials", None)
        else:
            sys.modules["local_operator.credentials"] = saved  # type: ignore[assignment]

    assert isinstance(snapshot, InfoSnapshot)
    assert snapshot.degraded


def test_a_degraded_collect_writes_nothing_into_the_current_directory(tmp_path: Path) -> None:
    """`/info` READS. It must never leave a file behind on the host.

    Found for real: the B1 fallback was `Path(".")`, and `CredentialManager`
    CREATES its store on construction, so probing a machine whose `config_dir()`
    could not be resolved wrote a `credentials.env` into whatever directory the
    user was standing in. A diagnostic that mutates the thing it describes is
    the same fault as `check_latest()` rewriting the cache, which this module
    bans outright.
    """
    import os

    import local_operator.paths as paths_mod

    def boom(*args: object, **kwargs: object) -> Path:
        raise OSError("home unreadable")

    original = paths_mod.config_dir
    cwd = os.getcwd()
    os.chdir(tmp_path)
    paths_mod.config_dir = boom  # type: ignore[assignment]
    try:
        snapshot = collect_snapshot(LiveState())
    finally:
        paths_mod.config_dir = original  # type: ignore[assignment]
        os.chdir(cwd)

    assert snapshot.degraded, "the failure is reported"
    assert list(tmp_path.iterdir()) == [], f"wrote {[p.name for p in tmp_path.iterdir()]}"


def test_a_registry_that_returns_zero_on_a_missing_root_is_still_degraded() -> None:
    """Q8: a count read from an unresolvable config root is not a measurement.

    `AgentRegistry.list_agents` and `CredentialManager.list_credential_keys`
    both RAISE on the unreadable sentinel and were correctly caught. But
    `TeamRegistry.list_teams` walks a missing directory and returns 0, so on an
    unresolvable home `teams` rendered a plausible `0` with NO degraded entry
    naming it anywhere — the one field on the screen with no honest marker.

    Keyed on the sentinel rather than on the value, because 0 is a legitimate
    answer on a machine that genuinely has no teams; suppressing every zero
    would trade one lie for another.
    """
    import local_operator.paths as paths

    original = paths.config_dir

    def boom() -> Path:
        raise RuntimeError("no home")

    paths.config_dir = boom  # type: ignore[assignment]
    try:
        errors: list[tuple[str, str]] = []
        agents = collect_agents(LiveState(), errors)
    finally:
        paths.config_dir = original  # type: ignore[assignment]

    named = {name for name, _ in errors}
    assert "agents.teams" in named, f"teams must be named as unreadable, got {named}"
    assert "agents.profiles" in named
    # The value is still the dataclass default; it is the DEGRADED entry that
    # makes the screen render `—` instead of that default.
    assert agents.teams == 0


# -- the fleet tally ----------------------------------------------------------
#
# The operator's question is "how many agent trajectories are running on this
# machine", and the whole hazard is answering it with a confident number whose
# terms are silently missing. Every test below is about the difference between
# "reported zero" and "did not report".


def test_a_mixed_fleet_excludes_the_unreported_and_names_how_many() -> None:
    """An older runtime is UNREPORTED, never a zero.

    The normal state of this host is a mixed fleet — windows are replaced a few
    at a time — so this is the common case, not a transient. Folding a
    non-reporting session in as 0 would make the total look complete while
    silently dropping every subagent those windows are running.
    """
    records = [
        (_Record(pid=1, busy=True, subagents_running=3, subagents_queued=1), "live"),
        (_Record(pid=2, subagents_running=0, subagents_queued=0), "live"),
        (_OldRecord(3, "tui", "s", "c", "/tmp", "m", 0.0, 0.0), "live"),
    ]
    info = collect_sessions(scan=_scan(records), usage=_usage({}), now=0.0)
    assert info.subagents_reporting == 2
    assert info.subagents_unreported == 1
    assert info.fleet_subagents_running == 3
    assert info.fleet_subagents_queued == 1
    # pid 2 REPORTED zero and is counted as reporting; pid 3 did not report at
    # all. Collapsing the two is the defect this pair exists to catch.
    assert info.lines[1].subagents_running == 0
    assert info.lines[2].subagents_running is None


def test_an_all_older_fleet_is_not_a_confident_zero() -> None:
    """Nothing reported, so the total is 0 WITH the caveat's denominator set."""
    records = [
        (_OldRecord(1, "tui", "s", "c", "/tmp", "m", 0.0, 0.0), "live"),
        (_OldRecord(2, "tui", "s", "c", "/tmp", "m", 0.0, 0.0), "live"),
    ]
    info = collect_sessions(scan=_scan(records), usage=_usage({}), now=0.0)
    assert info.subagents_reporting == 0
    assert info.subagents_unreported == 2
    assert info.fleet_subagents_running == 0
    # The screen renders a caveat off this, which is what stops the zero above
    # reading as a measurement.
    assert collect_agents(LiveState(), [], info).cross_session_known is False


def test_a_wedged_runtime_contributes_its_last_counts_and_a_stale_one_does_not() -> None:
    """A quiet pid can still have children burning tokens; a dead one cannot.

    ``wedged`` means no heartbeat for 45 s — the process is still there. ``stale``
    means ``registry.scan`` just DELETED the record because the pid is gone.
    Dropping the wedged session would under-report exactly the broken fleet this
    screen is opened to explain.
    """
    records = [
        (_Record(pid=1, subagents_running=1), "live"),
        (_Record(pid=2, busy=True, subagents_running=2), "wedged"),
        (_Record(pid=3, subagents_running=9), "stale"),
    ]
    info = collect_sessions(scan=_scan(records), usage=_usage({}), now=0.0)
    assert info.fleet_subagents_running == 3, "wedged in, stale out"
    assert info.subagents_reporting == 2


def test_session_and_subagent_trajectories_are_disjoint_and_added() -> None:
    """A subagent never publishes a record, so the two sets cannot overlap.

    ``harness.subagent`` builds a bare ``Session`` with no registrant, so a
    child is never also a runtime. That is the invariant that makes the
    addition legal rather than a double count.
    """
    records = [
        (_Record(pid=1, busy=True, subagents_running=2), "live"),
        (_Record(pid=2, busy=True, subagents_running=0), "live"),
        (_Record(pid=3, busy=False, subagents_running=1), "live"),
    ]
    info = collect_sessions(scan=_scan(records), usage=_usage({}), now=0.0)
    assert info.fleet_session_trajectories == 2
    assert info.fleet_subagents_running == 3
    assert info.fleet_trajectories == 5


def test_cross_session_known_means_somebody_ELSE_reported() -> None:
    """Not "the feature is compiled in" — the note it gates claims knowledge of
    OTHER windows, and on a single-session host that claim is false."""
    solo = collect_sessions(
        scan=_scan([(_Record(pid=1, subagents_running=4), "live")]),
        usage=_usage({}),
        self_pid=1,
        now=0.0,
    )
    assert collect_agents(LiveState(), [], solo).cross_session_known is False

    pair = collect_sessions(
        scan=_scan(
            [
                (_Record(pid=1, subagents_running=4), "live"),
                (_Record(pid=2, subagents_running=0), "live"),
            ]
        ),
        usage=_usage({}),
        self_pid=1,
        now=0.0,
    )
    assert collect_agents(LiveState(), [], pair).cross_session_known is True


def test_nested_subagents_are_counted_once() -> None:
    """``nodes()`` is already the COMPLETE roster including every descendant.

    So the count is a flat ``len()`` over a filter. A recursive child walk added
    on top would count every node below depth 0 twice, which is the specific
    trap the roster's own docstring warns about.
    """

    class _Comms:
        def nodes(self) -> list[Any]:
            return [
                _Node(job_id="a", label="reviewer"),
                _Node(job_id="b", label="scout", parent_job_id="a"),
                _Node(job_id="c", label="coder", parent_job_id="b"),
            ]

    class _Session:
        # Runtime role (SessionProtocol). This fake stands in for an OWNER:
        # it carries no attached runtime, which is what the absent legacy
        # `is_remote` meant.
        owns_runtime = True
        outcome_is_synchronous = True
        runtime_locality: RuntimeLocality = "this-process"

        subagent_comms = _Comms()

    live = collect_live(_Session())
    assert live.running == 3, "one per roster entry, not per path through the tree"
    assert len(live.tree) == 3


def test_a_follower_session_reports_its_own_subagents() -> None:
    """The §1.3 regression: a ``RemoteSession``-backed window showed ZERO.

    ``collect_live`` reads the PUBLIC ``subagent_comms``; ``RemoteSession`` only
    held ``_subagent_comms``, so the branch was skipped, the counts stayed 0 and
    nothing was recorded as degraded. Every attached window is on this path, so
    the screen said "no subagents" on a host full of them.

    Asserts the COUNT, not merely that rows exist: the naive fix (adding
    ``nodes()`` without projecting ``status``) makes the tree appear while the
    tally stays 0 and every row renders ``unknown``.
    """
    from local_operator.session.frontend_state import JobState, SnapshotSubagentComms
    from local_operator.session.remote import RemoteSession

    assert hasattr(RemoteSession, "subagent_comms"), "the public name is the contract"

    comms = SnapshotSubagentComms(
        [
            JobState(id="j1", type="task", status="running", label="reviewer"),
            JobState(id="j2", type="task", status="running", label="scout", parent_job_id="j1"),
            JobState(id="j3", type="task", status="queued", label="qa"),
            JobState(id="j4", type="task", status="completed", label="coder"),
        ]
    )
    follower = RemoteSession.__new__(RemoteSession)
    follower._subagent_comms = comms

    live = collect_live(follower)
    assert live.running == 2, f"the follower must report 2 running, got {live.running}"
    assert live.queued == 1
    assert live.settled == 1
    assert live.roster_unread is False
    assert [row.status for row in live.tree].count("running") == 2
    assert all(row.status for row in live.tree), "a node with no status renders as 'unknown'"


def test_a_session_with_no_comms_facade_is_degraded_not_zero() -> None:
    """The silent-default failure: nothing raised, so nothing was named.

    ``_safe`` catches probes that RAISE. This one returned a default and was
    believed, which is how the screen came to state "no subagents have been
    launched" about a session it had never actually asked.
    """

    class _Opaque:
        pass

    live = collect_live(_Opaque())
    assert live.roster_unread is True
    assert "live.subagents" in {name for name, _ in live.errors}
    # And the flag reaches the snapshot the renderer reads.
    snapshot = collect_snapshot(live, root=Path("/nonexistent-info-root"))
    assert snapshot.agents.roster_unread is True
    assert "live.subagents" in {name for name, _ in snapshot.degraded}


def test_a_comms_object_without_nodes_degrades_instead_of_raising() -> None:
    """``collect_live`` runs on the PAINT PATH and must never raise.

    The guard was ``_safe("live.subagents", comms.nodes, ...)``, which resolves
    the attribute as an ARGUMENT — before ``_safe`` enters its try — so a comms
    object lacking the method took the app down rather than degrading.
    """

    class _NoNodes:
        pass

    class _Session:
        # Runtime role (SessionProtocol). This fake stands in for an OWNER:
        # it carries no attached runtime, which is what the absent legacy
        # `is_remote` meant.
        owns_runtime = True
        outcome_is_synchronous = True
        runtime_locality: RuntimeLocality = "this-process"

        subagent_comms = _NoNodes()

    live = collect_live(_Session())  # must not raise
    assert live.roster_unread is True
    assert live.running == 0 and live.tree == ()
    assert "live.subagents" in {name for name, _ in live.errors}


def test_no_session_at_all_is_not_reported_as_a_failed_roster() -> None:
    """The CLI path passes ``session=None``. That is not a failed probe."""
    live = collect_live(None)
    assert live.roster_unread is False
    assert live.errors == ()


def test_a_wrong_typed_count_is_unreported_not_a_lost_section() -> None:
    """Q1 — one corrupt record used to blank the ENTIRE sessions block.

    ``SessionRecord.from_json`` filters keys and calls the constructor with no
    type validation, so these two fields — the first this module does
    ARITHMETIC on — are whatever the writer put in the file. A ``str`` or
    ``list`` raised ``TypeError`` inside the roll-up, and ``_safe`` guards whole
    SECTIONS, so a single bad record cost the session table, both fleet rows and
    the lower-bound caveat on the screen that exists for an already-broken host.
    ``main`` listed the same record fine, so it was a regression, not a limit.

    An unusable value is not a measurement: it is reported as ``None``, which
    the caveat already knows how to describe.
    """
    # Deliberately ill-typed: the whole point is a record this process did not
    # write, so the annotations are bypassed exactly as a bad JSON file does.
    bad_values: list[Any] = ["2", [2], {"n": 2}, object()]
    for bad in bad_values:
        records = [
            (_Record(pid=1, subagents_running=2, subagents_queued=0), "live"),
            (_Record(pid=2, subagents_running=bad, subagents_queued=bad), "live"),
        ]
        info = collect_sessions(scan=_scan(records), usage=_usage({}), now=0.0)
        assert info.available is True, f"{bad!r} must not cost the section"
        assert info.total == 2 and info.live == 2
        assert info.lines[1].subagents_running is None
        assert info.subagents_reporting == 1
        assert info.subagents_unreported == 1, "the bad record folds into the caveat"
        assert info.fleet_subagents_running == 2, "only the good record contributes"


def test_a_numerically_plausible_but_wrong_count_is_also_unreported() -> None:
    """Q1's quieter half: these never raised, they printed as fact.

    A float rendered ``4.5 total — 1 sessions + 3.5 subagents`` and a negative
    rendered ``-1 subagents``. Both are worse than the crash, because nothing
    signals that the number is not a measurement. ``bool`` is checked
    explicitly: it is an ``int`` subclass, so ``True`` would otherwise be
    counted as one subagent.
    """
    bad_values: list[Any] = [3.5, -1, True, False]
    for bad in bad_values:
        records = [(_Record(pid=1, subagents_running=bad, subagents_queued=bad), "live")]
        info = collect_sessions(scan=_scan(records), usage=_usage({}), now=0.0)
        assert info.lines[0].subagents_running is None, f"{bad!r} is not a count"
        assert info.subagents_unreported == 1
        assert info.fleet_subagents_running == 0
        assert isinstance(info.fleet_trajectories, int)

    # A genuine zero still reports, which is the distinction being protected.
    good = [(_Record(pid=1, subagents_running=0, subagents_queued=0), "live")]
    ok = collect_sessions(scan=_scan(good), usage=_usage({}), now=0.0)
    assert ok.lines[0].subagents_running == 0
    assert ok.subagents_reporting == 1 and ok.subagents_unreported == 0


def test_a_pausing_child_sorts_with_the_running_ones() -> None:
    """The tree must not contradict the tally directly above it.

    ``pausing`` is in ``RUNNING_SUBAGENT_STATUSES``, so the header counts it as
    a running trajectory. A local literal in ``rank()`` omitted it and sorted it
    below the settled rows — the same local-vs-shared divergence this module's
    own comments say the predicate exists to remove.
    """

    class _Node:
        def __init__(self, job_id: str, label: str, status: str) -> None:
            self.job_id = job_id
            self.label = label
            self.status = status
            self.parent_job_id = None
            self.agent_role = ""
            self.effort = ""
            self.session_id = ""

    rows, _, _ = build_subagent_tree(
        [
            _Node("a", "alpha", "completed"),
            _Node("b", "bravo", "pausing"),
            _Node("c", "charlie", "running"),
        ]
    )
    assert [row.label for row in rows] == ["bravo", "charlie", "alpha"]


def test_an_absurd_count_is_refused_rather_than_breaking_the_layout() -> None:
    """Q7: a 31-digit count renders 81 cells wide and overflows every frame.

    Not reachable from this codebase's publisher — the count is a ``len()`` over
    a roster bounded by ``DEFAULT_MAX_RUNNING_JOBS`` — so this only guards a
    corrupt or foreign record, which is exactly the population ``_reported_count``
    already exists to reason about. Six digits still pass: the ceiling is about
    rendering, not about a belief regarding how many subagents can exist.
    """
    for value, reported in ((999_999, True), (1_000_000, False), (10**30, False)):
        records = [
            (_Record(pid=1, subagents_running=1, subagents_queued=0), "live"),
            (_Record(pid=2, subagents_running=value, subagents_queued=0), "live"),
        ]
        info = collect_sessions(scan=_scan(records), usage=_usage({}), now=0.0)
        assert info.available is True
        assert (info.lines[1].subagents_running is not None) is reported, value
        assert info.subagents_unreported == (0 if reported else 1), value

"""A session moves ITSELF only when a real successor boots where it asked.

The cwd-move contract has two processes in it: the runtime that ASKS (its tool
call writes a durable intent and returns ``queued``) and the SUCCESSOR that
ACTS, by booting in the destination. These cells are the real ones — the
production ``process.py`` booted as a child, launched exactly as
``launch._spawn_runtime`` launches it — because the two properties they pin are
invisible to a fake:

* THE COMMIT IS THE SUCCESSOR'S, and it arrives at BOOT. Nothing a viewer, a
  frame or a stop can do commits the move early, and nothing about the request
  itself commits it at all.
* THE CONFIRMATION IS THE SESSION'S DIRECTORY, not this process's. A spawned
  runtime inherits the engaging host's working directory (``launch._spawn_runtime``
  passes no ``cwd=``) and carries the directory it really uses in
  ``LOP_MOBILE_CHILD_CWD``, so a boot reconcile that read ``os.getcwd()`` would
  compare the host's directory, never match, and leave every move ``queued``
  forever. That is what the second assertion in each cell exists to catch — the
  successor's own record names the destination while the intent is still
  ``queued``.

Isolation: ``headless_tui_env`` redirects the config dir and the root conftest
redirects ``HOME``; the child's environment is rebuilt with every ``CMUX_*``/
``LOP_MOBILE_CHILD_*``/``LOP_RUNTIME_*`` removed (a child that inherited a
workspace id could address the operator's live window, #648) and then given the
two names this contract actually uses. ``PYTHONPATH`` points at THIS checkout:
locally the venv's editable install resolves to the primary tree, and a child
that imported that instead would exercise code no PR here changes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

import pytest

from local_operator.session import cwd_move
from local_operator.session.runtime import registry
from tests.e2e.harness import NO_NOTIFY_ENV

pytestmark = pytest.mark.e2e

#: The session this cell moves. Fixed rather than generated: the marker, the
#: transcript and the child's ``--resume`` all key off it, and a readable id is
#: what a failing run's log needs.
SESSION_ID = "ownmove000001"

#: The inherited families a spawned runtime must never see. The same list the
#: cut-off e2e suite strips, for the same reason; ``LOP_BUILD_PREFIX`` is a bare
#: name whose consumers test it with ``startswith``.
_CHILD_ENV_FAMILIES = ("CMUX_", "LOP_MOBILE_CHILD_", "LOP_RUNTIME_", "LOP_BUILD_PREFIX")

#: A real boot takes ~20-25 s on this host under fleet load and the commit lands
#: while ``_serve`` binds, so the bound is generous on purpose: a shorter one
#: would fail for scheduling reasons and read as a broken feature.
BOOT_COMMIT_TIMEOUT_S = 120.0

#: The checkout under test, handed to the child so it imports THIS tree (see the
#: module docstring). ``parents[2]`` is the repo root for ``tests/e2e/<file>``.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])

_T = TypeVar("_T")


def _seed(config_dir: Path, session_id: str) -> Path:
    """A resumable session on the mock provider, with approvals OFF.

    ``tool_approval_mode: auto`` is the setting this contract is exercised under:
    the approval decision happens at the TOOL call (``move`` is a write-tier op),
    and these cells are about what happens after an intent exists. A gated
    session would park on an approval card nobody is present to answer.
    """
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    return directory


def _child_env(config_dir: Path, session_cwd: str) -> dict[str, str]:
    """The environment a cold child gets, with this contract's two names set."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_CHILD_ENV_FAMILIES)}
    env.update(NO_NOTIFY_ENV)
    env.update(
        {
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
            # THE DIRECTORY THE SESSION USES — the spawn contract, not the
            # process's cwd. See ``launch._spawn_runtime``: it sets this name and
            # passes no ``cwd=``, so the child inherits the engaging host's
            # directory. Passing the destination here is exactly what a viewer
            # that adopted the retiring frame does.
            "LOP_MOBILE_CHILD_CWD": session_cwd,
            "LOP_MOBILE_CHILD_RESUME": SESSION_ID,
            # Never the quiet exit: these cells kill one child on purpose.
            "LOP_SESSION_GRACE_S": "600",
            "PYTHONPATH": _REPO_ROOT,
        }
    )
    return env


def _spawn_successor(
    config_dir: Path, session_cwd: str, *, inherit_cwd: Path
) -> subprocess.Popen[bytes]:
    """Boot the REAL runtime the way a viewer's re-engage boots it.

    ``cwd=inherit_cwd`` is this cell's stand-in for "the directory the engaging
    host stood in" — deliberately NOT the session's: reproducing the production
    split is the whole point of the assertion that follows.
    """
    log = config_dir / "logs" / f"successor-{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as sink:
        return subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [sys.executable, "-m", "local_operator.session.runtime.process"],
            env=_child_env(config_dir, session_cwd),
            cwd=str(inherit_cwd),
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _reap(child: subprocess.Popen[bytes]) -> None:
    """Kill the child's whole process group by exact pid and wait for it.

    A real runtime owns a lease and a journal; leaving one running would hold the
    session against the NEXT cell in this file.
    """
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover — a wedged child is a failure
        pytest.fail(f"successor {child.pid} survived SIGKILL")


def _wait_until(predicate: Callable[[], _T | None], *, timeout: float, what: str) -> _T:
    """Poll ``predicate`` until it answers something truthy, or fail by name."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = predicate()
        if found is not None:
            return found
        time.sleep(0.5)
    pytest.fail(f"timed out after {timeout:.0f}s waiting for {what}")


def _record_cwd(config_dir: Path, pid: int) -> str | None:
    """The cwd the runtime ``pid`` PUBLISHED, or ``None`` while it has none.

    Keyed by PID, not by session: a killed predecessor leaves its record behind
    until something sweeps it, and a cell that asked "is any record for this
    session in the destination" would read the DEAD one and call the successor
    wrong (measured — the first run of this cell failed exactly that way).
    """
    for record, _state in registry.scan(config_dir):
        if getattr(record, "pid", None) == pid:
            return str(getattr(record, "cwd", "") or "") or None
    return None


def _stage(config_dir: Path) -> str:
    intent = cwd_move.read_move_intent(config_dir, SESSION_ID)
    return intent.stage if intent is not None else "missing"


def test_a_real_successor_boot_commits_the_move_and_nothing_else_does(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """The request is not the commit; the SUCCESSOR BOOTING is.

    Ordering is the contract. Asking writes the intent and changes nothing —
    including that the session keeps running where it is — so the first third of
    this cell watches that stage sit at ``queued`` with no successor alive at
    all. The commit then has to arrive from a real boot, and it has to arrive
    with the successor's own record naming the destination: that pair is what
    separates "the session confirmed" from "somebody's cwd happened to match".
    """
    config_dir = headless_tui_env
    _seed(config_dir, SESSION_ID)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    host_cwd = tmp_path / "host"
    for directory in (source, destination, host_cwd):
        directory.mkdir()

    intent = cwd_move.enqueue_self_move(
        config_dir,
        SESSION_ID,
        requested_cwd=str(destination),
        effective_cwd=str(source),
        owner_pid=os.getpid(),
    )
    assert intent.stage == cwd_move.STAGE_QUEUED
    # NOTHING ELSE COMMITS IT: three seconds of a real clock with no successor
    # anywhere, and the intent is still exactly where the request left it.
    time.sleep(3)
    assert _stage(config_dir) == cwd_move.STAGE_QUEUED
    assert not any(
        getattr(record, "session_id", "") == SESSION_ID
        for record, _state in registry.scan(config_dir)
    ), "no runtime for this session was started"

    child = _spawn_successor(config_dir, intent.destination_cwd, inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_COMMITTED or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor's boot to commit the move",
        )
        assert (
            _record_cwd(config_dir, child.pid) == intent.destination_cwd
        ), "the commit has to be the session that is really in the destination"
        stored = cwd_move.read_move_intent(config_dir, SESSION_ID)
        assert stored is not None
        assert stored.request_id == intent.request_id
        assert stored.source_cwd == intent.source_cwd
        assert stored.destination_cwd == intent.destination_cwd
    finally:
        _reap(child)


def test_a_crash_between_the_request_and_the_boot_is_settled_by_where_it_really_is(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """A lost confirmation is reconciled, and a wrong directory claims nothing.

    The intent outlives the process that wrote it — that is why it is a document
    rather than memory — so the first half kills its writer (SIGKILL, no clean
    exit) and asserts the request survived. The second half is the direction that
    must NOT be inferred: a successor that boots somewhere else leaves the intent
    ``queued``, because "not there yet" is not "failed", and a runtime that
    marked it failed would discard a request the user made.
    """
    config_dir = headless_tui_env
    _seed(config_dir, SESSION_ID)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    elsewhere = tmp_path / "elsewhere"
    host_cwd = tmp_path / "host"
    for directory in (source, destination, elsewhere, host_cwd):
        directory.mkdir()

    # 1. THE CRASH: a real process writes the intent and is killed uncleanly.
    writer = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            "-c",
            "import os, sys, time\n"
            "from pathlib import Path\n"
            "from local_operator.session import cwd_move\n"
            "cwd_move.enqueue_self_move(Path(sys.argv[1]), sys.argv[2],"
            " requested_cwd=sys.argv[3], effective_cwd=sys.argv[4], owner_pid=os.getpid())\n"
            "print('written', flush=True)\n"
            "time.sleep(300)\n",
            str(config_dir),
            SESSION_ID,
            str(destination),
            str(source),
        ],
        env={**os.environ, "PYTHONPATH": _REPO_ROOT},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        assert writer.stdout is not None
        assert writer.stdout.readline().strip() == b"written"
    finally:
        _reap(writer)
    crashed = cwd_move.read_move_intent(config_dir, SESSION_ID)
    assert crashed is not None, "the request survived the process that made it"
    assert crashed.stage == cwd_move.STAGE_QUEUED

    # 2. A SUCCESSOR IN THE WRONG DIRECTORY claims nothing — and, by design, it also
    #    RETIRES: its move watcher reads the queued intent at the first idle tick and
    #    stops the runtime, because confirming the move is the job of a successor that
    #    is IN the destination. So the observable here is its EXIT, not a published
    #    record: CI caught the earlier version waiting for a record this runtime may
    #    never write (it retires before the publisher runs on a fast host), which made
    #    a real behaviour read as a hang.
    wrong = _spawn_successor(config_dir, str(elsewhere), inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: True if wrong.poll() is not None else None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the wrong-directory successor to retire instead of claiming the move",
        )
        assert (
            _stage(config_dir) == cwd_move.STAGE_QUEUED
        ), "a boot elsewhere is not a confirmation, and not a failure either"
    finally:
        _reap(wrong)

    # 3. THE SUCCESSOR THAT REALLY IS IN THE DESTINATION settles it.
    right = _spawn_successor(config_dir, crashed.destination_cwd, inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_COMMITTED or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor in the destination to commit the reconciled intent",
        )
        _wait_until(
            lambda: _record_cwd(config_dir, right.pid) == crashed.destination_cwd or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor in the destination to publish its record there",
        )
    finally:
        _reap(right)

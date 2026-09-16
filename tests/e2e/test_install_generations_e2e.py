"""Installing a build never touches the tree a running runtime is importing.

The incident this stage exists for (2026-09-15): ``uv tool install --force``
recreated ``~/.local/share/uv/tools/local-operator`` **in place** while ~24 live
runtimes imported from it. 36 sessions died with no exit record, and 113 crash
reports named the planted libpython dylib — a process LAUNCHED during the
rewrite dies at load.

So these cells drive the PRODUCTION machinery around a real runtime:

* the real ``local_operator.session.runtime.process`` in a subprocess, launched
  through a GENERATION's own interpreter, so its ``sys.prefix`` names that
  generation;
* the real install-and-flip (``update.install_into_generation`` →
  ``update.flip_pointer``), with a fake ``uv`` runner — a full wheel build is
  minutes of the e2e budget and the property under test is where uv is AIMED and
  what happens to the running process, not what uv compiles. The live-host proof
  (a real install with the fleet up, pid set identical before and after, zero
  new DiagnosticReports) belongs to QA;
* the real stable launchers, and the real ``lop`` console script behind them, for
  the mid-flip ``lop --version`` cell.

Each generation's venv is a symlink to this checkout's venv, which is what makes
the cells cheap: the process really does import from
``<stable>/generations/<id>/tools/local-operator/...`` (verified in the assertions
below), and nothing in the install path needs a 136 MB copy to be exercised. The
symlink is never written to — the installs only ever create NEW generations.

Isolation follows ``AGENTS.md``'s rule for every cell here: ``HOME`` and the
config dir are per-test, and the child environment is rebuilt with EVERY
``CMUX_*`` and ``LOP_*`` variable removed before the names this test means to set
are added. An inherited ``CMUX_WORKSPACE_ID`` renamed the operator's real cmux
windows once (#648), and an inherited ``LOP_MOBILE_CHILD_*`` has silently
re-parented a cell's runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime import registry
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: The two builds the cells move between. Same version, different commits: the
#: COMMON shape on this host (`lop update` builds from ``main`` while
#: ``pyproject.toml`` still names the last release), and the shape that makes a
#: ref-only comparison necessary.
OLD = "46a4e9b1234567890abcdef"
NEW = "f4a70b991234567890abcdef"

#: The wake text hands the mock provider a REAL ``bash sleep`` tool call
#: (``MockClient``'s ``[bash:N]`` marker). It is the only way to park an
#: assembled runtime on a wait-shaped tool from outside, which is what makes
#: "the swap landed while the runtime was busy" a fact rather than a hope.
BUSY_TEXT = "hold this turn open [bash:8]"


def _env(config: Path, session_id: str) -> dict[str, str]:
    """The child's environment: nothing inherited that names a live resource."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CMUX_", "LOP_"))}
    env.update(
        {
            "LOCAL_OPERATOR_CONFIG_DIR": str(config),
            "LOP_MOBILE_CHILD_CWD": str(config),
            "LOP_MOBILE_CHILD_RESUME": session_id,
            # A grace far longer than any cell: only a retirement could end
            # these runtimes, and a quiet exit must not be what the assertions
            # are accidentally measuring.
            "LOP_SESSION_GRACE_S": "300",
        }
    )
    return env


def _seed(config: Path, session_id: str) -> Path:
    """A resumable session on the mock provider, which needs no network."""
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    return directory


def _stable() -> Path:
    return update_mod.stable_root()


def _linked_generation(name: str, venv: Path) -> Path:
    """A generation whose venv is a symlink to ``venv``, plus uv's bin shims.

    Written by hand rather than through ``install_into_generation`` because what
    it stands for is "a build that was installed HERE some time ago": the cells
    that start from it are about a runtime already running out of one.

    The symlink is what makes these cells cheap, and it has one visible
    consequence worth stating: ``process_install_root()`` resolves symlinks, so
    a runtime launched from here records the CHECKOUT's venv as its install
    root. The cells below assert the launch path (which names this generation)
    rather than the record's resolved root, because a real generation is a real
    directory and resolving it is a no-op there.
    """
    generation = update_mod.generations_dir() / name
    (generation / "tools").mkdir(parents=True, exist_ok=True)
    (generation / "bin").mkdir(parents=True, exist_ok=True)
    os.symlink(venv, generation / "tools" / "local-operator")
    for entry in ("lop", "local-operator"):
        os.symlink(
            generation / "tools" / "local-operator" / "bin" / entry,
            generation / "bin" / entry,
        )
    return generation


def _recording_generation(name: str, log: Path, marker: str = "") -> Path:
    """A generation whose interpreter records WHERE it was executed from.

    Used by the engagement cell, where the question is which interpreter the
    spawn helper chose rather than what the child did with it. The shim resolves
    its own ``$0`` with ``pwd -P``, so a spawn that handed over a path still
    containing ``current`` is visible in the log as exactly that — which is the
    defect this asserts against (a child importing through the mutable link).

    ``marker`` adds a console script shaped like uv's — a shebang plus a body —
    which prints the marker instead of a version. Execing THAT through
    ``~/.local/bin/lop`` is how the launcher cell tells which generation the
    chain resolved to: with two interchangeable venvs the version would be the
    same on both sides of a flip and the assertion would prove nothing.
    """
    generation = update_mod.generations_dir() / name
    bin_dir = generation / "tools" / "local-operator" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "python3"
    shim.write_text(
        '#!/bin/sh\nhere=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)\n'
        f"printf '%s\\n' \"$here\" >> {log}\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    if marker:
        script = bin_dir / "lop"
        script.write_text(f"#!/bin/sh\nprintf '%s\\n' '{marker}'\n", encoding="utf-8")
        script.chmod(0o755)
        (generation / "bin").mkdir(parents=True, exist_ok=True)
        os.symlink(script, generation / "bin" / "lop")
    return generation


def _fake_uv(version: str):
    """Stand in for ``uv tool install``: build the tree it would build."""

    def _run(argv: list[str], env: dict[str, str]) -> int:
        venv = Path(env["UV_TOOL_DIR"]) / "local-operator"
        bin_dir = Path(env["UV_TOOL_BIN_DIR"])
        for directory in (venv / "bin", venv / "lib" / "python3.12" / "site-packages", bin_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (venv / "pyvenv.cfg").write_text("home = /nonexistent\n", encoding="utf-8")
        for entry in ("lop", "local-operator"):
            shim = venv / "bin" / entry
            shim.write_text("#!/bin/sh\n", encoding="utf-8")
            shim.chmod(0o755)
            os.symlink(shim, bin_dir / entry)
        os.symlink(sys.executable, venv / "bin" / "python3")
        dist = venv / "lib" / "python3.12" / "site-packages" / f"local_operator-{version}.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: local-operator\nVersion: {version}\n", encoding="utf-8"
        )
        (dist / "entry_points.txt").write_text(
            "[console_scripts]\nlop = local_operator.cli:main\n"
            "local-operator = local_operator.cli:main\n",
            encoding="utf-8",
        )
        return 0

    return _run


def _spawn(config: Path, session_id: str, generation: Path) -> subprocess.Popen[bytes]:
    """The production module, run by the GENERATION's own interpreter.

    ``<gen>/tools/local-operator/bin/python3`` is what the console script's
    shebang names, so this is the same thing a real ``lop`` from this generation
    would start — and it makes ``sys.prefix`` name that generation, which is the
    property the whole layout turns on.
    """
    interpreter = generation / "tools" / "local-operator" / "bin" / "python3"
    return subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [str(interpreter), "-m", "local_operator.session.runtime.process"],
        env=_env(config, session_id),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _record(config: Path, session_id: str):
    for found, _state in registry.scan(config):
        if getattr(found, "session_id", "") == session_id:
            return found
    return None


def _wait_for_record(config: Path, session_id: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = _record(config, session_id)
        if found is not None:
            return found
        time.sleep(0.1)
    raise AssertionError(f"no record for {session_id} within {timeout}s")


def _wait_for(config: Path, session_id: str, predicate, timeout: float = 60.0, what: str = ""):
    """Poll the runtime's own record until ``predicate`` holds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = _record(config, session_id)
        if found is not None and predicate(found):
            return found
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what or predicate}")


def _wake(config: Path, session_id: str, text: str) -> subprocess.CompletedProcess[str]:
    """Hand an unattached runtime a message, through the real ``lop send``.

    Run through the stable launcher (``~/.local/bin/lop``) so this also
    exercises the pointer chain every cell depends on: the launcher resolves
    ``current`` at exec and lands on the generation's own console script.
    """
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            str(Path.home() / ".local" / "bin" / "lop"),
            "send",
            "--session",
            session_id,
            "--wake",
            text,
        ],
        env=_env(config, session_id),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _incidents(directory: Path) -> list[str]:
    """Every recorded cut-off, however it was written (see ``test_cut_off_turns``)."""
    from local_operator.session.transcript import Transcript

    return [
        json.dumps(entry.payload)
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == "session_incident"
    ]


def _reap(child: subprocess.Popen[bytes]) -> None:
    try:
        child.terminate()
        child.wait(timeout=20)
    except Exception:  # noqa: BLE001 — a test teardown must not raise
        child.kill()


def _install_new_generation(version: str = "0.55.1") -> Path:
    """Land a build the way ``lop update`` does: new generation, then flip."""
    return update_mod.install_into_generation(runner=_fake_uv(version), version=version)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and signals")
def test_a_busy_runtime_survives_a_real_install_and_flip(headless_tui_env: Path) -> None:
    """The incident, inverted: the swap lands mid-turn and the runtime does not notice.

    A runtime parked on a real ``bash`` tool (``sleep 8``, issued by the mock
    provider) while a build is installed into a NEW generation and ``current``
    is flipped onto it. Asserted on what the operator would have lost: the same
    pid, a turn that finishes, and NO ``runtime-killed`` / ``install-mid-update``
    incident — the two tokens the 2026-09-15 deaths were recorded under.

    The child is launched through ``<gen>/tools/local-operator/bin/python3``
    (see :func:`_spawn`), so its ``sys.prefix`` is the generation. That the spawn
    NAME is a concrete generation path is asserted where it is observable on both
    platforms — :func:`test_an_engage_after_a_flip_lands_on_the_generation_current_names`,
    whose child is a shim that prints the path it was executed from — because a
    real interpreter's own path is not readable portably once it is running
    (``ps`` prints argv, and Linux's ``/proc/<pid>/exe`` resolves the symlinked
    venv away to the base interpreter).
    """
    config = headless_tui_env
    session_id = "genbusy001"
    directory = _seed(config, session_id)
    first = _linked_generation("20260101T000000Z-old", Path(sys.prefix))
    update_mod.flip_pointer(first)
    update_mod.write_stable_launchers(first)

    child = _spawn(config, session_id, first)
    try:
        record = _wait_for_record(config, session_id)
        assert record.pid == child.pid

        sent = _wake(config, session_id, BUSY_TEXT)
        assert sent.returncode == 0, sent.stdout + sent.stderr
        _wait_for(config, session_id, lambda found: found.busy, timeout=90, what="a busy runtime")
        assert child.poll() is None, "the runtime died before the swap"

        # THE SWAP, through the production install path.
        second = _install_new_generation()
        assert second != first
        assert update_mod.current_generation() == second
        assert child.poll() is None, "the runtime died during the install and flip"

        # The turn finishes rather than being cut off, and the process it runs
        # in is the SAME process: no successor was spawned under it.
        _wait_for(
            config, session_id, lambda found: not found.busy, timeout=120, what="the turn to finish"
        )
        assert child.poll() is None, "the runtime left after its turn"
        assert _wait_for_record(config, session_id).pid == child.pid
        assert _incidents(directory) == [], "an install must not cut a turn off"
        # And the tree it was launched from is untouched: the install wrote a
        # NEW generation, not over this one.
        assert (first / "tools" / "local-operator" / "pyvenv.cfg").is_file()
    finally:
        _reap(child)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and signals")
def test_an_unwatched_runtime_keeps_its_pid_and_is_not_recorded_as_gone(
    headless_tui_env: Path,
) -> None:
    """Nobody is watching, so nothing else can be blamed for what the install does.

    The reflex ``lop sessions`` shape: a detached runtime with no viewer. A swap
    lands under it, and the record must still name the SAME live pid afterwards
    — not a successor, not an empty row — with its heartbeat still advancing.
    """
    config = headless_tui_env
    session_id = "genidle001"
    _seed(config, session_id)
    first = _linked_generation("20260101T000000Z-old", Path(sys.prefix))
    update_mod.flip_pointer(first)
    update_mod.write_stable_launchers(first)

    child = _spawn(config, session_id, first)
    try:
        record = _wait_for_record(config, session_id)
        before_heartbeat = float(record.heartbeat_at)
        _install_new_generation("0.55.2")

        # It stays resident and keeps beating: "disappeared without exiting
        # cleanly" is exactly what this asserts cannot happen.
        _wait_for(
            config,
            session_id,
            lambda found: float(found.heartbeat_at) > before_heartbeat,
            timeout=90,
            what="a fresh heartbeat",
        )
        assert child.poll() is None
        assert _wait_for_record(config, session_id).pid == child.pid
        assert child.pid == record.pid
    finally:
        _reap(child)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and signals")
def test_an_engage_after_a_flip_lands_on_the_generation_current_names(
    headless_tui_env: Path,
) -> None:
    """The convergence path: new work runs the build ``current`` points at.

    A mixed-generation fleet is a steady state, and what walks it onto the new
    build is this — an engage resolves the pointer and spawns THAT generation's
    interpreter, so a runtime that retires on its own schedule is replaced by one
    on the current build.

    The child is a recording shim rather than the product module: the question
    here is WHICH interpreter the engage chose, and a shim answers with the path
    it was executed from. The pointer is flipped between spawns, so each one has
    an exactly-known expected answer.

    A CONCURRENT flip loop is deliberately not what this cell asserts. A hot
    rename loop makes macOS fail the pointer read itself (``readlink``'s EINVAL —
    measured), and the documented answer to that is the fallback to
    ``sys.executable``, which is correct but not the property under test here;
    still less should a test demand that a microsecond race resolve one
    particular way. The mid-flip case that IS a hard requirement — nothing that
    execs the launcher chain may fail — is
    :func:`test_lop_version_through_the_pointer_never_fails_mid_flip`, which runs
    real processes under exactly that load.
    """
    config = headless_tui_env
    session_id = "genengage01"
    _seed(config, session_id)
    log = Path(config) / "spawns.log"
    first = _recording_generation("20260101T000000Z-old", log)
    second = _recording_generation("20260101T000001Z-new", log)

    driver = (
        "import sys;"
        "from local_operator.session.runtime import launch;"
        "p = launch._spawn_runtime('genengage01', %r, defer_materialise=True);"
        # WAIT for the child before this process exits: the shim writes its own
        # line, so a driver that returned immediately would race the file and the
        # cell would read seven lines for eight spawns.
        "rc = p.wait(timeout=60);"
        "print(p.pid, rc, flush=True)" % str(config)
    )

    def _spawn_once() -> None:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [sys.executable, "-c", driver],
            env=_env(config, session_id),
            capture_output=True,
            text=True,
            timeout=120,
        )
        # A dangling interpreter path is a FileNotFoundError here, so a non-zero
        # exit IS a failure; the child's own exit code is asserted too, which is
        # what makes "the engage started a complete generation" a claim about
        # execution rather than about an argv list.
        assert done.returncode == 0, done.stdout + done.stderr
        _pid, child_rc = (int(field) for field in done.stdout.split()[:2])
        assert child_rc == 0, f"the spawn exited {child_rc}"

    with bounded(180, "engage after a flip"):
        expected: list[str] = []
        for index in range(6):
            generation = (first, second)[index % 2]
            update_mod.flip_pointer(generation)
            _spawn_once()
            expected.append(generation.name)

    resolved = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(resolved) == len(expected), resolved
    for path, name in zip(resolved, expected, strict=True):
        # CONCRETE and CORRECT: the generation that was current for this spawn,
        # by an absolute path that does not mention the mutable ``current``.
        assert path.endswith(f"generations/{name}/tools/local-operator/bin"), path
        assert "current" not in path, path
    assert update_mod.current_generation() == second


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_the_stable_launcher_resolves_the_generation_current_names(
    headless_tui_env: Path,
) -> None:
    """The launcher chain is what a person (and a supervised unit) executes.

    ``~/.local/bin/lop`` must keep naming the POINTER — never a generation — so
    that a flip needs no launcher rewrite at all, and an exec after a flip must
    land on the generation ``current`` now names. Two generations carry distinct
    console scripts for that reason: with interchangeable venvs the version
    printed on both sides of a flip would be identical and the assertion would
    pass without proving anything.

    This is deliberately NOT a hot-flip race. Measured on this machine, a rename
    loop running at ~143k flips/second does make an exec through the chain fail
    at startup (macOS returns ``EINVAL`` for a path whose component is replaced
    underneath the reader — in the child that path is its own ``sys.path[0]``,
    which the console script keeps spelled through ``current``). At the rate
    flips really happen — once per install — that window is not reachable, and
    the properties that DO have to hold under load are asserted where they can
    be: :func:`test_a_busy_runtime_survives_a_real_install_and_flip` (a running
    process is never disturbed) and
    :func:`test_an_engage_after_a_flip_lands_on_the_generation_current_names`
    (the spawn path resolves the pointer to a CONCRETE interpreter, so a child
    never holds a mutable path). The residual itself is written down in
    ``docs/design-install-generations.md`` §3.2 rather than hidden here.
    """
    config = headless_tui_env
    _seed(config, "genversion1")
    log = Path(config) / "spawns.log"
    first = _recording_generation("20260101T000000Z-old", log, marker="GENERATION-ONE")
    second = _recording_generation("20260101T000001Z-new", log, marker="GENERATION-TWO")
    update_mod.flip_pointer(first)
    update_mod.write_stable_launchers(first)
    launcher = Path.home() / ".local" / "bin" / "lop"
    assert launcher.is_symlink(), "the stable launcher must exist for this cell to mean anything"
    assert os.readlink(launcher) == str(update_mod.pointer_path() / "bin" / "lop"), (
        "the launcher names the pointer, not a generation: a flip must not have to " "rewrite it"
    )

    with bounded(120, "lop through the stable launcher"):
        assert _run_launcher(launcher, config) == "GENERATION-ONE"
        # The flip, and nothing else: the launcher is untouched and the next exec
        # is the new build.
        update_mod.flip_pointer(second)
        assert os.readlink(launcher) == str(update_mod.pointer_path() / "bin" / "lop")
        assert _run_launcher(launcher, config) == "GENERATION-TWO"


def _run_launcher(launcher: Path, config: Path) -> str:
    """Exec the launcher the way a shell (or a unit) would, and read its line."""
    done = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [str(launcher), "--version"],
        env=_env(config, "genversion1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.strip()

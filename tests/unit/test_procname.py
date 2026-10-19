"""Guards for process attribution — the branded interpreter image.

WHAT IS ACTUALLY AT RISK HERE, and therefore what these tests pin:

1. **The staleness rule.** The planted link is a hardlink, so it pins an
   INODE, not a version. Every trigger in :func:`procname._needs_replant`
   corresponds to a way the shape silently rots into "a stale interpreter under
   fresh site-packages" or "aborts on every launch". A trigger that stops
   firing is invisible in review and in production until someone's `lop` breaks.
2. **The fallback ladder.** This whole feature is decoration on a process
   listing. Every rung must be a silent no-op, because the alternative — a
   startup that fails over its own cosmetics — is far worse than an
   unrecognisable row in Activity Monitor.
3. **The plists.** macOS names a background login item by the basename of
   ``ProgramArguments[0]``, which is why installing a daemon used to announce
   "python3 is running in the background".

The real end-to-end behaviour (that a process executed through the link
actually reports ``proc_name = b'Local Operator'`` to the kernel) is verified
on the PR with live ``ps``/``proc_name`` captures: it cannot be asserted here
without planting links on the machine running the suite.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import procname
from local_operator.interpreter import SAFE_PATH_FLAG

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the branded-image mechanism is macOS-specific; Linux uses prctl",
)


def _fake_venv(tmp_path: Path) -> Path:
    """A venv-shaped tree that BELIEVES it is a venv.

    Real, because the thing under test IS filesystem behaviour: inode identity,
    link counts and symlink resolution cannot be faked with a mock and still
    prove anything. ``pyvenv.cfg`` is part of that shape rather than decoration:
    the runtime image has none beside it and CPython locates the stdlib
    differently in the two cases, so a fixture without it would exercise the
    image path while claiming to be a venv.
    """
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
    real = Path(os.path.realpath(sys.executable))
    (tmp_path / "pyvenv.cfg").write_text(
        f"home = {real.parent}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    return tmp_path


def _generation_venv(root: Path) -> Path:
    """A SECOND venv-shaped tree, as a ``lop`` generation looks to a spawn.

    Returns its ``bin/python3``. Four choices in here are deliberate:

    * ``bin/python3`` is a SYMLINK to the versioned binary — which is what every
      generation venv has, and the whole bug: the kernel resolves a symlink,
      takes ``p_comm`` from the target, and the process is ``python3.x``.
    * the interpreter it points at is a COPY, in a SEPARATE base tree, rather
      than another hardlink to the real one — so the inode this test plants
      against, and any branded child it executes, belongs to a throwaway tree.
      Measured reason: a remediation an EDR applies to a flagged branded file
      reaches the SHARED interpreter through a hardlink, and on 2026-09-19 that
      took out uv's 3.12.13 and 3.13.12 on the operator's machine (see
      ``docs/ENDPOINT_PROTECTION.md``). A test run must not be able to do that to
      the fleet's interpreters.
    * the base tree is SEPARATE from the venv because that is production's shape
      (a generation venv's ``python3`` points into a shared interpreter tree),
      and it is what makes the companion-dylib claim testable at all (review
      round 1, R-2): the plant's SOURCE is the base tree's ``lib`` and its
      TARGET is the venv's, so a missing or wrong venv-side pin is a state the
      plant has to repair. With source and target in one directory they coincide
      and no assertion can tell the plant from the fixture — mutation
      ``_plant_libpython -> True`` passed every cell then.
    * ``<venv>/lib`` starts EMPTY: nothing is pre-planted, so the symlink the
      cells assert on is the plant's own work.
    """
    root.mkdir(parents=True, exist_ok=True)
    base = _generation_base(root)
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "lib").mkdir(parents=True, exist_ok=True)
    (base / "bin").mkdir(parents=True, exist_ok=True)
    (base / "lib").mkdir(parents=True, exist_ok=True)
    real = Path(os.path.realpath(sys.executable))
    reason = _unbrandable_reason(root)
    if reason is not None:
        pytest.skip(reason)
    dylib = real.parent.parent / "lib" / f"lib{real.name}.dylib"
    if not dylib.is_file():
        pytest.skip(f"no {dylib.name} beside {real}: the copy could not be executed")
    (root / "pyvenv.cfg").write_text(
        f"home = {base / 'bin'}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    shutil.copy(real, base / "bin" / real.name)
    # The BASE tree's pin is the plant's source; the venv's is the plant's job.
    os.symlink(dylib, base / "lib" / dylib.name)
    target = root / "bin" / "python3"
    os.symlink(Path("..") / ".." / base.name / "bin" / real.name, target)
    return target


def _runnable_venv(root: Path) -> Path:
    """A generation-shaped venv whose interpreter can actually START.

    Returns its ``bin/python3``. The base tree is a copy of this interpreter
    with symlinks into the real ``lib`` (stdlib and dylib), which is what
    ``_generation_venv`` deliberately does not provide: that fixture is for
    tests that never execute the target, and a child exec'd from it dies with
    ``No module named 'encodings'``. Nothing here touches uv's tree — whose
    mtime is part of the build key for every venv on the machine.
    """
    real = Path(os.path.realpath(sys.executable))
    if ".framework" in str(real):
        pytest.skip("framework interpreters cannot be branded (stub re-exec)")
    base = root / "base"
    (base / "bin").mkdir(parents=True)
    (base / "lib").mkdir(parents=True)
    copied = base / "bin" / real.name
    shutil.copy2(real, copied)
    for entry in (real.parent.parent / "lib").iterdir():
        if entry.name.startswith("python") or entry.name.startswith("libpython"):
            os.symlink(entry, base / "lib" / entry.name)
    venv = root / "gen"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n", encoding="utf-8")
    (venv / "bin" / "python3").symlink_to(copied)
    return venv / "bin" / "python3"


def _generation_base(root: Path) -> Path:
    """The shared interpreter tree a generation venv's ``bin/python3`` points into.

    Spelled once because the fixture creates it and the companion-dylib
    assertion names it as the plant's SOURCE; two spellings would let the test
    prove a different directory from the one the fixture built.
    """
    return root.parent / f"{root.name}-base"


def _unbrandable_reason(root: Path) -> str | None:
    """Why a plant into ``root`` cannot be attempted, or ``None``.

    The same two refusals the ``branded`` fixture checks, spelled once: a
    framework interpreter (whose stub re-execs and discards the link's name)
    and a cross-device tmp dir (``os.link`` cannot cross filesystems). Both are
    environment properties, not test failures.
    """
    real = Path(os.path.realpath(sys.executable))
    if ".framework" in str(real):
        return "framework interpreters cannot be branded (stub re-exec)"
    if real.stat().st_dev != os.stat(root).st_dev:
        return "interpreter and tmp_path on different devices: hardlink impossible"
    return None


@pytest.fixture(autouse=True)
def stable_runtime(tmp_path, monkeypatch):
    """Keep every plant out of the operator's real ``~/.local/share/lop``.

    ``LOCAL_OPERATOR_RUNTIME_ROOT`` rather than a monkeypatched function: the
    plants here are re-checked by CHILD processes (a spawn executes the link),
    and an environment variable is the only override that crosses that
    boundary. See :func:`procname.runtime_root` for why the variable exists.
    """
    root = tmp_path / "runtime"
    monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(root))
    return root


@pytest.fixture()
def branded(tmp_path, monkeypatch, stable_runtime):
    """Point ``procname`` at a throwaway prefix and plant into it.

    ``sys.prefix``/``sys.base_prefix`` are patched so the module believes it is
    in a venv it owns, and the runtime root is redirected so the image is
    published under ``tmp_path``. NOTHING is written outside ``tmp_path``: the
    operator's real venvs and installs must never be touched by a test run.
    """
    prefix = _fake_venv(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    real = Path(os.path.realpath(sys.executable))
    if ".framework" in str(real):
        pytest.skip("framework interpreters cannot be branded (stub re-exec)")
    if real.stat().st_dev != os.stat(prefix).st_dev:
        pytest.skip("interpreter and tmp_path on different devices: no clone")
    link = procname.ensure_branded_interpreter()
    if link is None:
        pytest.skip("no branded image could be planted in this environment")
    return link


@pytest.fixture()
def hardlink_branded(branded, monkeypatch):
    """The FALLBACK rung: a runtime root that cannot be planted into.

    The per-venv hardlink and its dylib pin are one release of insurance
    (decision D2), and every one of their staleness rules still has a caller:
    a read-only or full stable root, an interpreter whose tree cannot back an
    image. (A non-APFS volume is not one: the clone falls back to a byte copy
    and keeps the runtime rung.) Faking the rung unavailable is how those rules are
    exercised without a read-only filesystem.
    """
    # Start from the per-venv shape this rung exists to serve. ``branded`` has
    # already planted the runtime link, and leaving it there would make
    # ``ensure`` (correctly) keep it: a healthy runtime link is never replaced
    # by the fallback.
    real = Path(os.path.realpath(sys.executable))
    branded.unlink()
    os.link(real, branded)
    monkeypatch.setattr(procname, "_plant_runtime", lambda real, root: None)
    link = procname.ensure_branded_interpreter()
    assert link is not None, "the fallback rung must still plant"
    assert not link.is_symlink(), "this fixture is the hardlink rung by definition"
    return link


class TestStaleness:
    """The FALLBACK rung's triggers, each demonstrated firing AND repairing.

    The runtime rung's own staleness rule is identity against the build image
    (``TestStableRuntimeImage``); these are the inode-shaped rules the per-venv
    hardlink keeps for one release of insurance (decision D2). They are
    exercised through ``hardlink_branded``, which plants with the runtime rung
    unavailable — the same state as a read-only stable root.
    """

    def test_replants_when_link_names_a_different_inode(self, hardlink_branded, tmp_path):
        """Trigger (a): ``uv tool install --force`` replaced the interpreter.

        uv PRESERVES unknown files in ``bin/`` but does not refresh them, so the
        planted link keeps naming the OLD inode. Left unrepaired the process
        runs a stale interpreter against fresh site-packages.

        The stand-in is deliberately a file with ``st_nlink >= 2``. A plain
        ``write_bytes`` here has ``nlink == 1``, so trigger (b) fires as well
        and the test passes even when the inode comparison is deleted —
        verified by mutation: removing trigger (a) left this test green until
        the link count was made innocent.
        """
        real_ino = os.stat(os.path.realpath(sys.executable)).st_ino
        stale = tmp_path / "previous-interpreter"
        stale.write_bytes(b"the interpreter uv just replaced")
        hardlink_branded.unlink()
        os.link(stale, hardlink_branded)  # different inode, but nlink == 2
        assert os.stat(hardlink_branded).st_ino != real_ino
        assert os.stat(hardlink_branded).st_nlink >= 2, "only trigger (a) may detect this"

        assert procname.ensure_branded_interpreter() == hardlink_branded
        assert os.stat(hardlink_branded).st_ino == real_ino

    def test_replants_when_hardlink_became_a_copy(self, branded):
        """Trigger (b): ``st_nlink < 2``.

        An archive restore or an ``rsync`` without ``-H`` turns the hardlink
        into a copy. A copy is a different inode AND cannot find
        ``@rpath/libpython`` — the abort mode again.
        """
        branded.unlink()
        branded.write_bytes(Path(os.path.realpath(sys.executable)).read_bytes())
        assert os.stat(branded).st_nlink == 1

        assert procname.ensure_branded_interpreter() == branded
        assert branded.is_symlink(), "a copy at the path is replaced by the image link"

    def test_replants_when_libpython_symlink_dangles(self, hardlink_branded):
        """Trigger (c): the measured 100%-abort mode.

        Without ``<venv>/lib/libpython3.X.dylib`` the branded hardlink dies with
        ``Abort trap: 6`` on every launch — 40/40 measured. A first run can
        spuriously succeed from a warm dyld launch closure, which is exactly why
        this is pinned by a test rather than by a manual smoke check.
        """
        name = procname._libpython_name()
        assert name, "a brandable interpreter must expose an LDLIBRARY dylib"
        lib = hardlink_branded.parent.parent / "lib" / name
        assert lib.is_symlink() and lib.exists(), "plant must create a live symlink"

        lib.unlink()
        lib.symlink_to("/nonexistent/libpython.dylib")
        assert not lib.exists(), "precondition: the symlink dangles"

        assert procname.ensure_branded_interpreter() == hardlink_branded
        assert lib.exists(), "the dangling symlink must be repaired"

    def test_orphan_temps_from_dead_plants_are_swept(self, hardlink_branded):
        """A plant killed between ``os.link`` and ``os.replace`` leaks a temp.

        Reproduced with ``kill -9``: the cleanup on the error path cannot run,
        so one entry accumulates per crashed startup. A LIVE pid's temp must
        survive — a concurrent worktree is plausibly mid-plant.
        """
        bin_dir = hardlink_branded.parent
        dead = bin_dir / f".{procname.BRAND}.999999.tmp"
        # A DIFFERENT live pid: this process's own temp name is the one the
        # plant itself claims and consumes via `os.replace`, so using it here
        # would test the plant's own bookkeeping rather than the sweep.
        mine = bin_dir / f".{procname.BRAND}.{os.getppid()}.tmp"
        dead.write_bytes(b"orphan")
        mine.write_bytes(b"in flight")

        hardlink_branded.unlink()  # force the replant path, which is where the sweep runs
        assert procname.ensure_branded_interpreter() == hardlink_branded

        assert not dead.exists(), "a dead pid's temp must be swept"
        assert mine.exists(), "a live pid's temp must be left alone"

    def test_the_orphan_sweep_asks_procstate_and_never_signals(self, hardlink_branded, monkeypatch):
        """The sweep asks ``procstate.pid_alive``; it must never touch ``os.kill``.

        THE C1-CLASS TRAP: ``os.kill(pid, 0)`` is a liveness probe on POSIX and
        ``TerminateProcess`` on Windows, so a sweep written on it would KILL
        whichever live process matched a stale temp entry's embedded pid — then
        read the successful kill as "still running" and keep the entry as well.
        This runs on every startup, so the blast radius is "any live process".

        Pinned as a FORBIDDEN CALL rather than behind a platform branch: making
        the signal fatal fails this test on macOS too, which is what stops the
        probe being reintroduced on the platform where nobody would notice.
        """
        from local_operator import procstate

        def _forbidden(pid: int, sig: int) -> None:  # pragma: no cover - the guard
            raise AssertionError("the orphan sweep must not signal: os.kill is a KILL on Windows")

        monkeypatch.setattr(os, "kill", _forbidden)
        asked: list[int] = []

        def _dead(pid: int) -> bool:
            asked.append(pid)
            return False

        monkeypatch.setattr(procstate, "pid_alive", _dead)

        orphan = hardlink_branded.parent / f".{procname.BRAND}.999999.tmp"
        orphan.write_bytes(b"orphan")
        hardlink_branded.unlink()  # force the replant path, which is where the sweep runs

        assert procname.ensure_branded_interpreter() == hardlink_branded

        assert 999999 in asked, "the sweep must ask the shared liveness probe"
        assert not orphan.exists()

    def test_importing_procname_defers_procstate(self) -> None:
        """MINOR 2 (review round 4): the sweep's import stays inside the sweep.

        ``procname`` is imported on the startup path of every ``lop`` invocation, and
        the ``procstate`` import inside :func:`close_orphaned_temps` is deliberate:
        measured, ``import local_operator.procname`` leaves ``procstate`` out of
        ``sys.modules``. (What it does NOT save is ``subprocess`` — ``cli.py`` imports
        that itself — so the deferral buys one stdlib-only module body, which is the
        whole reason the comment above it says so.) Nothing pinned that locality, so
        restoring the module-level form — the line a merge resolution dropped — read
        like a no-op cleanup while quietly adding a module body to every invocation.

        THE PROBE RUNS IN A CHILD, like the sibling spawn/import probes above, and for
        the same reason: this test module and the suite's other imports may already
        have pulled ``procstate`` in, so an in-process ``sys.modules`` check would
        assert about collection order rather than about ``procname``.
        """
        probe = (
            "import sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parents[2])
            + "import local_operator.procname;"
            "print('local_operator.procstate' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False", (
            "importing procname must not pull procstate in: the common startup path "
            "does not need it, and the sweep imports it for itself"
        )

    def test_replants_when_libpython_points_at_the_wrong_library(self, hardlink_branded):
        """Trigger (c), second half: a LIVE symlink is not automatically CORRECT.

        Found while verifying the B4 fix. `exists()` only proves the link
        resolves, so after an interpreter upgrade a link aimed at the PREVIOUS
        install's dylib passed the check — and loading a mismatched libpython
        beside a fresh interpreter is the crash this shape exists to avoid.
        """
        name = procname._libpython_name()
        assert name
        lib = hardlink_branded.parent.parent / "lib" / name
        lib.unlink()
        lib.symlink_to("/etc/hosts")  # live, resolvable, and wrong
        assert lib.exists(), "precondition: the symlink resolves"

        assert procname.ensure_branded_interpreter() == hardlink_branded
        expected = Path(os.path.realpath(sys.executable)).parent.parent / "lib" / name
        assert lib.resolve() == expected.resolve()

    def test_orphan_temps_are_swept_in_lib_as_well_as_bin(self, hardlink_branded):
        """REGRESSION (review round 1, B3): both plant directories are swept.

        A killed plant leaks a temp named for the BRAND in `bin/` or for the
        DYLIB in `lib/`. Sweeping only `bin/` left lib temps accumulating
        forever, one per crashed startup.
        """
        name = procname._libpython_name()
        assert name
        bin_orphan = hardlink_branded.parent / f".{procname.BRAND}.999999.tmp"
        lib_orphan = hardlink_branded.parent.parent / "lib" / f".{name}.999998.tmp"
        bin_orphan.write_bytes(b"orphan")
        lib_orphan.write_bytes(b"orphan")

        hardlink_branded.unlink()  # force the replant path
        assert procname.ensure_branded_interpreter() == hardlink_branded

        assert not bin_orphan.exists(), "bin/ orphan must be swept"
        assert not lib_orphan.exists(), "lib/ orphan must be swept too"

    def test_replanting_onto_an_identical_link_consumes_its_temp(self, hardlink_branded):
        """REGRESSION: the ordinary replant must not leak its staging name.

        Trigger (c) — the missing companion libpython — fires while ``hardlink_branded``
        ALREADY names the right inode, so the plant's ``os.replace(tmp, link)``
        moves one hardlink onto another hardlink of the SAME file. POSIX says
        that call "shall return successfully and perform no other action", so
        the temp was never consumed: measured on this host as 573 leftovers
        across 17 generations, one per stale-fleet startup, and the reproduced
        shape below is a single plant call leaving one behind.

        This is checked through the real entry point, because that is where the
        leak reaches the operator's ``<venv>/bin`` — and because the sweep that
        runs immediately before the plant cannot mask it: the temp carries THIS
        process's pid, which is alive.
        """
        real = Path(os.path.realpath(sys.executable))
        assert os.stat(hardlink_branded).st_ino == os.stat(real).st_ino, "precondition: same inode"
        name = procname._libpython_name()
        assert name
        lib = hardlink_branded.parent.parent / "lib" / name
        lib.unlink()
        lib.symlink_to("/nonexistent/libpython.dylib")  # trigger (c), and only (c)

        assert procname.ensure_branded_interpreter() == hardlink_branded

        leftovers = sorted(
            path.name for path in hardlink_branded.parent.glob(f".{procname.BRAND}.*.tmp")
        )
        assert leftovers == [], f"the replant leaked its temp: {leftovers}"
        assert hardlink_branded.exists(), "the link the caller asked for must survive"

    def test_orphan_temp_of_a_zombie_planter_is_swept(self, hardlink_branded):
        """A SIGKILLed plant is a ZOMBIE, not a live pid, and its temp is litter.

        ``os.kill(pid, 0)`` answers for an exited-but-unreaped process exactly
        as it does for a running one — measured here on a killed child that
        ``ps`` reports as ``Z``. The sweep therefore kept every temp of every
        crashed plant for as long as the parent failed to reap it, which is
        precisely the fleet-wide case it exists for: when 25 runtimes die at
        once, the parent that would reap them is the frontend that just lost
        them, so signal 0 goes on saying "alive" for all 25.

        A pid that is alive in any other state must still be left alone —
        ``test_orphan_temps_from_dead_plants_are_swept`` pins that half.
        """
        from local_operator import procstate

        # ``posix_spawn`` rather than ``os.fork``/``Popen``: this runs inside an
        # xdist worker (multi-threaded, so ``fork`` warns about deadlocks), and
        # the child must die WITHOUT the parent reaping it — which is exactly
        # what a Popen object cannot promise, since every poll/context-manager
        # exit reaps it. The pid is kept, never handed to anything that waits.
        pid = os.posix_spawn(sys.executable, [sys.executable, "-c", "pass"], dict(os.environ))
        try:
            # Wait on the EVENT (this pid is now a zombie), never on a clock.
            for _ in range(500):
                if procstate.is_zombie(pid):
                    break
                time.sleep(0.01)
            else:
                pytest.fail(f"pid {pid} never became a zombie")
            # Precondition for the bug rather than a restatement of it: the
            # probe the sweep used to rely on cannot tell a zombie from a
            # running plant, so this pid reads alive to it.
            os.kill(pid, 0)

            bin_dir = hardlink_branded.parent
            zombie_temp = bin_dir / f".{procname.BRAND}.{pid}.tmp"
            zombie_temp.write_bytes(b"orphan of a killed plant")

            hardlink_branded.unlink()  # force the replant path, which is where the sweep runs
            assert procname.ensure_branded_interpreter() == hardlink_branded

            assert not zombie_temp.exists(), "a zombie plant's temp must be swept"
        finally:
            os.waitpid(pid, 0)

    def test_unreadable_libpython_parent_does_not_raise(self, branded, tmp_path):
        """QA round 2, Q3: `_needs_replant` is contractually no-raise.

        An unreadable parent directory made `Path.exists()` raise
        `PermissionError` (reproduced on 3.12), which escaped a function this
        module documents as never raising. "Replant" is the safe answer: a
        libpython we cannot stat is not one to assume correct.

        The wall is asserted with `os.stat` rather than `Path.exists` because
        the stdlib changed underneath the original precondition: 3.14 routes
        `exists()` through `os.path.exists`/`_stat(ignore_errors=True)`, so
        EACCES becomes `False` instead of propagating (3.12 raises). The route
        this test is named for is still exercised wherever the stdlib still has
        it — `_needs_replant` calls `exists()` itself, so an unguarded raise
        there fails this test on 3.12 — while the property being pinned, "must
        not raise on a path it cannot stat", holds on every interpreter.
        """
        walled = tmp_path / "lib" / "sub"
        walled.mkdir(parents=True)
        dylib = walled / "libpython3.12.dylib"
        dylib.touch()
        walled.chmod(0o000)
        try:
            # PRECONDITION, not decoration. `True` is ALSO reachable through the
            # `resolve()` identity mismatch below, so asserting it alone would
            # not prove this test exercised the unreadable path it is named for
            # — and on a host where `chmod 0o000` does not block (root, some
            # container runtimes) it would stay green with the guard deleted.
            # Review round 3, M3-2: the R2-1 failure mode in embryo.
            with pytest.raises(PermissionError):
                os.stat(dylib)
            assert (
                procname._stale_hardlink(branded, Path(os.path.realpath(sys.executable)), dylib)
                is True
            )
        finally:
            # Restore so tmp_path teardown can remove the tree.
            walled.chmod(0o755)

    def test_healthy_shape_is_not_rewritten(self, branded):
        """The common path does no filesystem writes.

        This runs on every startup, so a plant that re-links unconditionally
        would churn the path (and race sibling worktrees) for nothing. The
        probe is ``lstat``: ``stat`` would follow the symlink to the shared
        image, which is the same for every venv and would hide a rewritten link.
        """
        before = os.lstat(branded)
        target = os.readlink(branded)
        assert procname.ensure_branded_interpreter() == branded
        after = os.lstat(branded)
        assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
        assert os.readlink(branded) == target

    def test_replants_when_the_link_is_present_but_not_executable(self, tmp_path):
        """Trigger (d), isolated from (a) and (b) so it cannot pass by accident.

        MEASURED, not imagined: an EDR on the operator's machine replaced the
        planted link in five worktree venvs with a 49968-byte, mode-0600 file
        owned by its own agent user (``_sentinel``), so the path named an image
        ``ds`` could neither read nor exec. The shape it replaced was otherwise
        correct.

        So the link here genuinely IS the interpreter's inode with
        ``st_nlink >= 2`` — the file it points at is a COPY of the interpreter,
        which is the only way to hold a non-executable inode without
        ``chmod``-ing the shared interpreter (see the module docstring: a mode
        write on the link changes the mode of every venv on the machine).
        Only the new exec probe can fire, and the positive control on the next
        line proves the same shape is otherwise considered healthy.
        """
        real = tmp_path / "bin" / "python3"
        real.parent.mkdir(parents=True)
        real.write_bytes(Path(os.path.realpath(sys.executable)).read_bytes())
        link = tmp_path / "bin" / procname.BRAND
        real.chmod(0o755)
        os.link(real, link)
        assert (
            procname._stale_hardlink(link, real, None) is False
        ), "the control: an executable inode-matching link is healthy"
        real.chmod(0o600)
        assert procname._stale_hardlink(link, real, None) is True

    def test_a_neutered_link_is_repaired_rather_than_handed_out(self, branded):
        """(d's) outcome: the plant replaces it, and the pair gets the new link.

        ``branded`` planted a healthy link; the file is then replaced with the
        measured EDR shape (a FOREIGN file, not the interpreter's inode, mode
        0600). Both halves matter: the next spawn must not be handed that file
        as ``executable=`` (it would die at ``execve``), and the repair must be
        a real one rather than a withheld name.
        """
        branded.unlink()
        branded.write_bytes(b"x" * 49968)
        branded.chmod(0o600)
        assert os.access(branded, os.X_OK) is False

        argv0, executable = procname.spawn_identity(procname.LABEL_EVAL, id="deadbeef")

        assert executable == str(branded)
        assert os.access(branded, os.X_OK) is True
        assert branded.is_symlink(), "the repair must plant the image link, not a copy"
        image = Path(os.readlink(branded))
        real = procname.real_interpreter()
        assert real is not None
        assert image.parent.parent.name == procname._build_key(
            real
        ), "the repair must name the image for THIS interpreter build"
        assert image.stat().st_ino != os.stat(os.path.realpath(sys.executable)).st_ino
        assert argv0 == "Local Operator [eval] session=deadbeef"

    def test_an_image_whose_stdlib_does_not_resolve_is_never_handed_out(self, branded):
        """The gate ``os.access`` cannot see: an executable image with no stdlib.

        A symlink to the runtime clone whose ``lib`` no longer resolves is
        executable and still dies at ``No module named 'encodings'``, so an
        image is only handed to ``subprocess`` when the stdlib resolves through
        the ``lib`` beside it. This is the runtime rung's equivalent of the
        no-dylib refusal the hardlink rung carries.
        """
        image = Path(os.readlink(branded))
        (image.parent.parent / "lib").unlink()
        os.symlink("/nonexistent/lib", image.parent.parent / "lib")

        # A path INSIDE a venv is resolved through that venv's pyvenv.cfg, so
        # only the image itself (no pyvenv.cfg beside it) is asked the question.
        assert not (image.parent.parent / "pyvenv.cfg").exists()
        assert procname._executable_image(image) is None

    def test_a_corrupt_published_image_is_rebuilt(self, branded):
        """A published image that CANNOT start is not a race winner.

        Published directories are immutable by convention, not at the cost of
        a machine that can never repair itself: an image whose stdlib does not
        resolve cannot have a running process behind it, so it is removed and
        republished rather than silently downgrading every venv to the
        fallback rung for the life of the install.
        """
        image = Path(os.readlink(branded))
        (image.parent.parent / "lib").unlink()
        os.symlink("/nonexistent/lib", image.parent.parent / "lib")

        assert procname.ensure_branded_interpreter() == branded
        assert procname._executable_image(branded) is not None
        assert procname._stdlib_through(Path(os.readlink(branded)))

    def test_an_unrepairable_neutered_link_withholds_the_label_too(self, branded, monkeypatch):
        """The EDR-RACE case: the plant reports success and the file is still dead.

        A spawn that trusted the plant's return value would hand ``subprocess``
        an image that is not executable by us. Rung 2 is the contract here —
        the bare interpreter and NO label (a labelled ``argv[0]`` alongside no
        image is the row this whole change exists to remove).

        The plant is faked as succeeding while the file stays inert because
        that is literally what the machine does: the EDR rewrites the link
        after our ``os.replace``.
        """
        branded.unlink()
        branded.write_bytes(b"x" * 49968)
        branded.chmod(0o600)
        # The runtime rung is faked as unavailable so the path taken is the
        # one this test is about: a fallback plant that REPORTS success while
        # the file stays inert.
        monkeypatch.setattr(procname, "_plant_runtime", lambda real, root: None)
        monkeypatch.setattr(procname, "_plant_hardlink", lambda link, real: True)

        argv0, executable = procname.spawn_identity(procname.LABEL_EVAL, id="deadbeef")

        assert executable is None
        assert argv0 == sys.executable
        assert procname.branded_argv0(procname.LABEL_EVAL, id="deadbeef") not in argv0


class TestFallbackLadder:
    """Every rung is a silent no-op. None of these may raise."""

    def test_non_venv_interpreter_is_never_planted_into(self, monkeypatch):
        """A system/Homebrew prefix is shared, possibly root-owned, and not ours."""
        monkeypatch.setattr(sys, "prefix", "/usr")
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        assert procname.branded_link_path() is None
        assert procname.ensure_branded_interpreter() is None

    def test_framework_build_is_refused(self, tmp_path, monkeypatch):
        """A framework interpreter re-execs an inner stub, discarding our name.

        Measured: hardlinking Homebrew's ``python@3.14`` and running it reports
        ``proc_name = b'Python'``. Planting there would cost an inode and buy
        nothing.
        """
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
        monkeypatch.setattr(
            procname,
            "_real_interpreter",
            lambda: Path("/opt/x/Python.framework/Versions/3.14/bin/python3.14"),
        )
        assert procname.branded_link_path() is None

    def test_unplantable_prefix_is_silent(self, tmp_path, monkeypatch):
        """A read-only or foreign-owned prefix yields None, never an exception."""
        monkeypatch.setattr(sys, "prefix", "/proc/nonexistent-prefix")
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path))
        assert procname.ensure_branded_interpreter() is None

    def test_a_source_that_cannot_back_an_image_plants_nothing(
        self, tmp_path, monkeypatch, stable_runtime
    ):
        """No stdlib in the source tree AND no dylib to pin: rung 2, honestly.

        The runtime image would die at ``No module named 'encodings'`` and the
        hardlink at dyld's missing dylib (the measured 100%-abort mode), so
        nothing is planted and the caller is told there is no name. The
        alternative — planting one of them blind — is a launch that fails
        everywhere the warm dyld closure is not.
        """
        monkeypatch.setattr(sys, "prefix", str(_fake_venv(tmp_path)))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
        monkeypatch.setattr(procname, "_stdlib_through", lambda image: False)
        monkeypatch.setattr(procname, "_libpython_name", lambda: None)
        assert procname.ensure_branded_interpreter() is None
        assert not (tmp_path / "bin" / procname.BRAND).exists()
        real = procname.real_interpreter()
        assert real is not None and not (stable_runtime / procname._build_key(real)).exists()

    def test_reexec_returns_when_branding_is_unavailable(self, monkeypatch):
        """``reexec_branded`` must RETURN (not exit) so the process stays alive.

        This is the property that lets the ``lop`` shebang keep pointing at the
        real interpreter: a failed branding leaves a live process that can
        repair the link, where a bad shebang would have exited 126 before any
        code ran.
        """
        monkeypatch.setattr(procname, "should_reexec", lambda: None)
        called: list[object] = []
        monkeypatch.setattr(os, "execv", lambda *a: called.append(a))
        procname.reexec_branded()
        assert called == []

    def test_in_process_main_never_replaces_its_caller(self, monkeypatch):
        """THE REGRESSION THIS GUARD EXISTS FOR.

        ``cli.main()`` is called in-process by this very suite (``assert
        main() == 7``). An unguarded re-exec there replaces the running pytest
        with a fresh interpreter: observed as a 98-test file silently
        truncating at 57% with exit code 0 — a green run that tested nothing.
        """
        # sys.orig_argv under pytest is not a launcher invocation.
        assert procname.is_own_launch() is False
        assert procname.should_reexec() is None

        called: list[object] = []
        monkeypatch.setattr(os, "execv", lambda *a: called.append(a))
        procname.reexec_branded("Local Operator [serve] port=1")
        assert called == [], "a non-launcher process must never exec itself"

    @pytest.mark.parametrize(
        ("orig_argv", "expected"),
        [
            (["python", "/x/bin/lop", "--resume", "a"], True),
            (["python", "/x/bin/lo"], True),
            (["python", "/x/bin/local-operator", "serve"], True),
            (["python", "-m", "local_operator.cli"], True),
            (["python", "-m", "local_operator"], True),
            (["python", "-m", "pytest", "tests/unit"], False),
            (["python", "-c", "import local_operator.cli"], False),
            (["python"], False),
            (["python", "/x/bin/some-other-tool"], False),
            # LEADING INTERPRETER OPTIONS ARE NOT THE PROGRAM. The first row is
            # OUR OWN reload successor — the argv `server/reload.py` execs when
            # it replaces a serving daemon (live pid 1343: `python3.14`, ppid 1,
            # `--listener-fd`) — and the second is a spawn of ours that answered
            # False before this change. The desktop app is NOT in this list and
            # must not be added: it launches its backend as `-c …` and probes
            # with the same string, which is the row two lines below that has to
            # stay False (review round 1, R-6).
            (["python", "-P", "-m", "local_operator.cli", "serve"], True),
            (["python", "-P", "-m", "local_operator.cli", "serve", "--port", "1"], True),
            (["python", "-u", "-m", "local_operator.tools.eval_worker"], True),
            (["python", "-P", "-u", "-E", "-s", "-I", "-B", "-m", "local_operator"], True),
            (["python", "-X", "dev", "-m", "local_operator.cli"], True),
            (["python", "-Xdev", "-m", "local_operator.cli"], True),
            (["python", "-W", "ignore", "-m", "local_operator.cli"], True),
            (["python", "-Wignore", "-m", "local_operator.cli"], True),
            (["python", "--check-hash-based-pycs=always", "-m", "local_operator.cli"], True),
            (["python", "-P", "/x/bin/lop"], True),
            # ANY submodule of the package, not just the two spellings the old
            # test listed: a spawn of our own module is a launch of this
            # product, which is why the eval worker and the runtime's own
            # module are in this list.
            (["python", "-m", "local_operator.tools.eval_worker"], True),
            (["python", "-m", "local_operator.session.runtime.process"], True),
            # ... and the two directions that must NOT move. The `-c` refusal is
            # a contract with the desktop app, not an oversight: its identity
            # probe is the same `-c` string asking for `sys.executable`, and a
            # branded answer makes the app refuse its own backend.
            (["python", "-P", "-c", "from local_operator.cli import main; main()"], False),
            (["python", "-P", "-u", "-c", "import local_operator.cli"], False),
            (["python", "-ccode"], False),
            (["python", "-P", "-m", "pytest", "tests/unit"], False),
            (["python", "-P", "-m"], False),
            (["python", "-P"], False),
            (["python", "-Q", "-m", "local_operator.cli"], False),
            # Bounded at the dot: a similarly-named distribution is not ours.
            (["python", "-m", "local_operators.cli"], False),
            (["python", "-m", "local_operator_x"], False),
        ],
    )
    def test_launch_detection(self, monkeypatch, orig_argv, expected):
        monkeypatch.setattr(sys, "orig_argv", orig_argv)
        assert procname.is_own_launch() is expected

    def test_reexec_is_suppressed_once_branded(self, monkeypatch, branded):
        """Already running through the link => no second exec. No loop.

        The loop guard is the process's OWN identity (`sys.executable` is the
        branded link), not an environment marker — see
        `test_reexec_leaves_the_environment_untouched` for why a marker was
        removed.

        THIS TEST MUST FAIL IF THE GUARD IS DELETED, and an earlier version did
        not (review round 2, R2-1): it patched `sys.executable` but left
        pytest's own `sys.orig_argv`, so `is_own_launch()` returned False and
        `should_reexec()` returned None at the NEXT early return — one line
        below the line under test. The assertion passed on a path that never
        reached the guard, and deleting the guard left 352 tests green.

        So every other reason to return None is neutralised first, and the
        preconditions are asserted rather than assumed: a REAL planted link
        (a `tmp_path` fake makes `ensure_branded_interpreter()` fail on a fake
        prefix, which is a second independent way to pass vacuously) and an
        `orig_argv` that `is_own_launch()` accepts. The guard is then the only
        thing left that can produce None.
        """
        link = branded  # the real planted link, not a fake
        monkeypatch.setattr(sys, "executable", str(link))
        monkeypatch.setattr(sys, "orig_argv", [str(link), "/x/.venv/bin/lop", "serve"])

        # Preconditions: without these the assertion below is vacuous.
        assert procname.is_own_launch() is True
        assert procname.ensure_branded_interpreter() is not None

        assert procname.should_reexec() is None, "a branded process must not re-exec again"

    def test_reexec_leaves_the_environment_untouched(self, monkeypatch):
        """REGRESSION (review round 1, B1+B2): no marker may enter os.environ.

        An earlier revision set `LOCAL_OPERATOR_BRANDED` in the LIVE
        environment before exec'ing. Two things broke, both reproduced:

        - `/reload` and `/update` (`reexec.replace_self`) do
          `env = os.environ.copy()` and exec the plain `lop` launcher, so the
          relaunch inherited "already branded" and the session was de-branded
          permanently (post-reload image: `python`);
        - every child spawned with an inherited environment (`launch.py` does
          `dict(os.environ)`) carried it, so genuine launches refused to brand.

        The environment this process hands to `exec` must therefore be exactly
        the one it already had.
        """
        before = dict(os.environ)
        captured: dict[str, object] = {}

        def _fake_execv(path, argv):
            captured["path"] = path
            captured["argv"] = argv

        monkeypatch.setattr(procname, "should_reexec", lambda: Path("/x/Local Operator"))
        monkeypatch.setattr(os, "execv", _fake_execv)
        # `execve` would also be a defect here: it is how a marker gets passed
        # without mutating os.environ, and this feature needs no marker at all.
        monkeypatch.setattr(
            os,
            "execve",
            lambda *a: pytest.fail("re-exec must not pass a modified environment"),
        )
        procname.reexec_branded("Local Operator [serve] port=1")

        assert captured["path"] == "/x/Local Operator"
        assert dict(os.environ) == before, "re-exec must not mutate the live environment"

    def test_reload_roundtrip_rebrands(self, branded):
        """A relaunch after `/reload` must brand AGAIN, not inherit a marker.

        Drives the real shape: a branded process (image = the link) spawns a
        child with `os.environ.copy()`, exactly as `reexec.replace_self` does.
        That child must see nothing that would stop it branding.
        """
        report = branded.parent.parent / "reload_probe.py"
        report.write_text(
            "import os, sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
            "from local_operator import procname\n"
            # The relaunch runs the ORDINARY interpreter (the lop launcher's
            # shebang), so it must not already look branded...
            "print(os.path.basename(sys.executable) == procname.BRAND)\n"
            # ...and nothing in the inherited environment may veto branding.
            "print(any('BRANDED' in k for k in os.environ))\n"
        )
        handoff = branded.parent.parent / "reload_handoff.py"
        handoff.write_text(
            "import os, subprocess, sys\n"
            f"subprocess.run([{str(os.path.realpath(sys.executable))!r}, {str(report)!r}],"
            " env=os.environ.copy(), check=True)\n"
        )
        result = subprocess.run(
            [procname.branded_argv0(procname.LABEL_SERVE, port=1), str(handoff)],
            executable=str(branded),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        looks_branded, has_marker = result.stdout.split()
        assert looks_branded == "False", "the relaunched interpreter must not look branded"
        assert has_marker == "False", "no inherited marker may suppress re-branding"

    def test_inherited_env_child_is_not_vetoed(self, branded):
        """REGRESSION (B2): nothing in an inherited environment suppresses branding.

        `launch.py` hands a runtime `dict(os.environ)` wholesale. When the
        parent wrote a marker into its own environment, that child — a genuine
        `lop` launch — inherited "already branded" and silently declined.

        The assertion is about the ENVIRONMENT, not about whether this
        particular child plants: a process exec'd from a link outside a venv
        reports `sys.prefix == sys.base_prefix` and correctly refuses to plant
        into a shared interpreter prefix. What must hold is that the decision
        is made on the child's own merits with no inherited veto.
        """
        probe = (
            "import os, sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parents[2])
            + "from local_operator import procname;"
            "print(any('BRANDED' in k for k in os.environ));"
            # The refusal, when it happens, is the venv check and nothing else.
            "print(sys.prefix != sys.base_prefix or procname.branded_link_path() is None)"
        )
        result = subprocess.run(
            [procname.branded_argv0(procname.LABEL_SESSION_ANON, id="deadbeef"), "-c", probe],
            executable=str(branded),
            env=dict(os.environ),  # exactly what launch.py hands a runtime
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        has_marker, decided_on_own_merits = result.stdout.split()
        assert has_marker == "False", "no marker may leak into an inherited environment"
        assert decided_on_own_merits == "True"

    def test_ensure_never_raises(self, monkeypatch):
        """Contract: decoration must never break a startup, whatever fails."""

        def boom() -> None:
            raise RuntimeError("filesystem on fire")

        monkeypatch.setattr(procname, "branded_link_path", boom)
        assert procname.ensure_branded_interpreter() is None
        procname.reexec_branded()  # must not raise either


class TestLabels:
    """argv is a PUBLIC channel; the vocabulary is fixed templates only."""

    def test_labels_render_with_the_brand(self):
        assert (
            procname.branded_argv0(procname.LABEL_SERVE, port=8080)
            == "Local Operator [serve] port=8080"
        )
        assert (
            procname.branded_argv0(procname.LABEL_EVAL, id="deadbeef")
            == "Local Operator [eval] session=deadbeef"
        )

    def test_agent_field_is_reduced_to_a_parseable_token(self):
        """The one label field that can carry human text stays a bounded token.

        An agent name with spaces or a newline would split the `[session]
        agent=…` row into something unreadable in `ps` and unmatchable by
        `pgrep -f`.
        """
        assert procname.safe_field("weird name; rm -rf /") == "weird-name--rm--rf--"
        assert procname.safe_field("a" * 99) == "a" * 24
        assert procname.safe_field("") == "-"
        assert procname.safe_field("coder") == "coder"

    def test_cli_sanitises_the_agent_before_labelling(self):
        from local_operator import cli

        args = cli.build_cli_parser().parse_args(["--agent", "my agent"])
        assert cli._process_label(args) == "Local Operator [session] agent=my-agent"

    def test_unrenderable_label_degrades_to_the_brand(self):
        """A spawn must never fail over its own decoration."""
        assert procname.branded_argv0(procname.LABEL_SERVE) == procname.BRAND

    def test_brand_fits_the_kernel_and_ps_windows(self):
        """``p_comm`` holds 31 chars; ``ps -o ucomm`` truncates at 16.

        The brand must be distinguishable within the SHORTER window, or every
        row in ``ps`` reads the same.
        """
        assert len(procname.BRAND) <= 31
        assert len(procname.BRAND) <= 16


class TestLaunchdPrograms:
    """All four LaunchAgents name their item 'Local Operator', not 'python3'."""

    @pytest.mark.parametrize(
        "module",
        [
            "local_operator.mobile.service",
            "local_operator.browser_bridge.daemon",
            "local_operator.tunnels.service",
            "local_operator.wakes.supervisor",
        ],
    )
    def test_program_arguments_head_is_the_branded_image(self, module, branded):
        program = procname.launchd_program(module, "--port", "1234")
        # BTM (the notification and System Settings > Login Items) reads the
        # BASENAME of element 0 — not Label, not the plist filename.
        assert os.path.basename(program[0]) == procname.BRAND
        assert program[1:] == ["-m", module, "--port", "1234"]

    def test_falls_back_to_sys_executable(self, monkeypatch):
        """Unbrandable environment => byte-for-byte the previous plist."""
        monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: None)
        assert procname.launchd_program("local_operator.mobile.service") == [
            sys.executable,
            "-m",
            "local_operator.mobile.service",
        ]

    def test_all_four_renderers_route_through_the_helper(self, monkeypatch, tmp_path):
        """Pins that no installer reverts to a bare ``sys.executable``.

        Each of these was the documented anti-pattern before this change; a
        revert would restore "python3 is running in the background" with no
        test failing anywhere else. Both axes are asserted: ``Program`` is the
        branded image, ``ProgramArguments[0]`` is that daemon's role label.
        """
        from local_operator.browser_bridge import install as browser_install
        from local_operator.mobile import install as mobile_install
        from local_operator.tunnels import install as tunnel_install
        from local_operator.wakes import install as wakes_install

        sentinel = tmp_path / "bin" / procname.BRAND
        sentinel.parent.mkdir(parents=True)
        sentinel.touch()
        monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: sentinel)

        # `render_plist` is typed `dict[str, object]`, so the element type is
        # narrowed here rather than indexed straight off an `object`.
        rendered = {
            "mobile": mobile_install.render_plist(1),
            "browser": browser_install.render_plist(1),
            "wakes": wakes_install.render_plist(tmp_path),
            "tunnel": tunnel_install.render_plist(),
        }
        first_rows = set()
        for name, plist in rendered.items():
            program = plist["ProgramArguments"]
            assert plist["Program"] == str(sentinel), name
            assert isinstance(program, list)
            assert program[0].startswith(procname.BRAND), (name, program[0])
            first_rows.add(program[0])
        # The point of the change: four daemons, four distinguishable rows.
        assert len(first_rows) == 4, first_rows


class TestSpawnIdentity:
    """``(argv[0], executable)`` — the pairing that makes every spawn named.

    A PAIR, and only a pair: on POSIX ``Popen(argv=[…])`` with
    ``executable=None`` EXECUTES ``argv[0]``, so a spawn site that decorated
    ``argv[0]`` with a label on its own would ask the kernel to run a file named
    ``Local Operator [eval] session=…``. The other half of the contract is the
    one CI taught: the label is applied ONLY alongside a branded image, because
    a labelled ``argv[0]`` empties the child's ``sys.executable`` on Linux (see
    ``tests/unit/test_spawn_naming_fallback.py``).
    """

    def test_pairs_the_label_with_the_branded_image(self, branded):
        argv0, executable = procname.spawn_identity(procname.LABEL_SESSION_ANON, id="abcd1234")
        assert argv0 == "Local Operator [session] id=abcd1234"
        assert executable == str(branded)

    def test_without_an_image_the_label_is_withheld(self, monkeypatch):
        """Rung 2, and the reason it is not "argv-only labelling".

        The row stays ``python3.x`` here. Labelling it would look better in
        ``ps`` and cost the child its interpreter identity on Linux, which is a
        trade this project does not make: see the module ladder and the
        Linux-executed child test in ``test_spawn_naming_fallback.py``.
        """
        monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: None)
        argv0, executable = procname.spawn_identity(procname.LABEL_EVAL, id="deadbeef")
        assert argv0 == sys.executable
        assert executable is None

    def test_the_pair_actually_runs(self, monkeypatch):
        """The pairing is EXECUTED, not merely returned — and the child is whole.

        Both halves at once: the argv/``executable=`` pair starts the
        interpreter, and the child that comes out of it can still say which
        interpreter it is (the property Linux loses to a label).
        """
        monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: None)
        argv0, executable = procname.spawn_identity(procname.LABEL_EVAL, id="deadbeef")
        result = subprocess.run(
            [argv0, "-c", "import sys; print(sys.executable or 'missing')"],
            executable=executable,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        # realpath, not string equality: a platform is free to report the image
        # it resolved, and the property under test is that the child HAS a
        # working interpreter rather than which spelling of it came back.
        assert os.path.realpath(result.stdout.strip()) == os.path.realpath(sys.executable)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX execve semantics")
    def test_a_label_without_an_image_cannot_be_executed(self):
        """The hazard the pairing exists for, demonstrated rather than assumed."""
        with pytest.raises(OSError):
            subprocess.run(
                [procname.branded_argv0(procname.LABEL_EVAL, id="deadbeef"), "-c", "pass"],
                timeout=30,
            )

    def test_a_failing_probe_withholds_the_label_too(self, monkeypatch):
        """No-raise applies to the probe too: a failed probe is rung 2, not a label."""

        def explode():
            raise RuntimeError("no stat")

        monkeypatch.setattr(procname, "ensure_branded_interpreter", explode)
        argv0, executable = procname.spawn_identity(procname.LABEL_SERVE, port=1)
        assert argv0 == sys.executable
        assert executable is None


class TestCrossTreeSpawnIdentity:
    """``spawn_identity_for_interpreter``: the pair for ANOTHER tree's build.

    WHAT WAS BROKEN, in the operator's own words on 2026-09-19: an EDR killed a
    Local Operator process 1079 times on a colleague's Mac. The threat name was
    ``python3.14`` and the flagged child's argv read ``Local Operator [daemons]
    refresh …`` — a LABEL ON AN UNBRANDED IMAGE, which is both the row an EDR
    keys on and a violation of this module's own rule that the label rides with
    the image or not at all. Both production sites that run a different tree's
    interpreter were doing exactly that.

    So the assertions here are about the PAIR, and the headline one is negative:
    a label must be impossible without an image we can execute.
    """

    def test_plants_beside_the_target_and_pairs_the_label(self, tmp_path):
        target = _generation_venv(tmp_path / "gen")
        link = target.parent / procname.BRAND

        argv0, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_DAEMONS_REFRESH, str(target)
        )

        assert argv0 == "Local Operator [daemons] refresh"
        assert executable == str(link)
        assert link.stat().st_ino == os.stat(os.path.realpath(target)).st_ino
        assert os.access(link, os.X_OK), "an image a spawn will exec must be executable"
        # The companion dylib pin is mandatory (the 100%-abort mode) and it must
        # be planted INTO THE TARGET'S venv, pointing at the TARGET's own dylib
        # rather than at ours. The fixture leaves `<venv>/lib` empty and puts the
        # source in a separate base tree precisely so this asserts the PLANT's
        # work: before that, the fixture pre-created the symlink and mutation
        # `_plant_libpython -> True` (a no-op plant) passed every cell here
        # (review round 1, R-2).
        pin = target.parent.parent / "lib" / f"lib{Path(os.path.realpath(target)).name}.dylib"
        assert pin.is_symlink(), f"the plant must create {pin}"
        assert pin.resolve() == (_generation_base(tmp_path / "gen") / "lib" / pin.name).resolve()

    def test_the_pair_runs_the_target_tree(self, tmp_path):
        """Executed, not merely returned — and the child is the TARGET's build.

        Two properties on one spawn, and the second is why the helper takes an
        interpreter at all: the name must come from the tree being executed,
        and the child must still report THAT tree's ``sys.prefix`` (a
        generation flip converges one session at a time through it).
        """
        target = _runnable_venv(tmp_path / "runnable")
        argv0, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_DAEMONS_REFRESH, str(target)
        )
        probe = (
            "import ctypes, os, sys;"
            "l=ctypes.CDLL('/usr/lib/libSystem.dylib');"
            "b=ctypes.create_string_buffer(64);"
            "l.proc_name(ctypes.c_int(os.getpid()),b,ctypes.c_uint(64));"
            "print(b.value.decode());"
            "print(sys.prefix)"
        )
        result = subprocess.run(
            [argv0, SAFE_PATH_FLAG, "-c", probe],
            executable=executable,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        name, prefix = result.stdout.strip().splitlines()[:2]
        assert name == procname.BRAND
        assert os.path.realpath(prefix) == os.path.realpath(target.parent.parent)

    def test_an_interpreter_with_no_venv_beside_it_is_rung_2(self, tmp_path):
        """A non-venv target is refused the same way the own-venv path refuses one."""
        root = tmp_path / "usr" / "bin"
        root.mkdir(parents=True)
        target = root / "python3"
        os.symlink(Path(os.path.realpath(sys.executable)), target)

        argv0, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_DAEMONS_REFRESH, str(target)
        )

        assert (argv0, executable) == (str(target), None)
        assert not (root / procname.BRAND).exists()

    def test_a_target_with_no_dylib_is_rung_2(self, tmp_path):
        """No dylib to pin means the hardlink would abort: refuse, never plant.

        A copy of the interpreter in a tree with an empty ``lib/`` is the
        cleanest real stand-in for the static-build case ``_libpython_name``
        already refuses for this venv. The interpreter is never executed, so
        its ``@rpath`` is irrelevant here.
        """
        target = _generation_venv(tmp_path / "gen")
        weird = tmp_path / "weird"
        (weird / "bin").mkdir(parents=True)
        (weird / "lib").mkdir(parents=True)
        (weird / "pyvenv.cfg").write_text("home = /nowhere\n", encoding="utf-8")
        copied = weird / "bin" / "python3.12"
        copied.write_bytes(Path(os.path.realpath(target)).read_bytes())

        argv0, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_DAEMONS_REFRESH, str(copied)
        )

        assert (argv0, executable) == (str(copied), None)

    def test_a_cross_device_target_is_rung_2(self, tmp_path, monkeypatch):
        """``EXDEV`` is an expected outcome, not an error to report (see the ladder)."""
        import errno

        target = _generation_venv(tmp_path / "gen")

        def cross_device(source, destination):
            raise OSError(errno.EXDEV, "Cross-device link")

        monkeypatch.setattr(procname.os, "link", cross_device)
        argv0, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_DAEMONS_REFRESH, str(target)
        )

        assert (argv0, executable) == (str(target), None)

    def test_a_neutered_link_is_replanted_not_handed_out(self, tmp_path):
        """The measured EDR shape, on the target side this time.

        The first call plants a healthy pair; the link is then replaced with a
        mode-0600 foreign file (different inode, not executable by us). The
        second call must repair it — and the pair it returns must name the
        REPAIRED link, never the dead one.
        """
        target = _generation_venv(tmp_path / "gen")
        link = target.parent / procname.BRAND
        procname.spawn_identity_for_interpreter(procname.LABEL_DAEMONS_REFRESH, str(target))
        link.unlink()
        link.write_bytes(b"x" * 49968)
        link.chmod(0o600)

        _, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_DAEMONS_REFRESH, str(target)
        )

        assert executable == str(link)
        assert os.access(link, os.X_OK) is True
        assert link.stat().st_ino == os.stat(os.path.realpath(target)).st_ino

    def test_an_unrepairable_target_link_withholds_the_label(self, tmp_path):
        """A read-only target ``bin/``: no repair is possible, so NO label.

        This is the invariant, not a corner case: the rung-2 pair is the bare
        target path, which still runs the right build, and no ``ps`` reader is
        told a name the process does not have.
        """
        target = _generation_venv(tmp_path / "gen")
        link = target.parent / procname.BRAND
        link.write_bytes(b"x" * 49968)
        link.chmod(0o600)
        target.parent.chmod(0o500)
        try:
            argv0, executable = procname.spawn_identity_for_interpreter(
                procname.LABEL_DAEMONS_REFRESH, str(target)
            )
        finally:
            target.parent.chmod(0o700)

        assert (argv0, executable) == (str(target), None)
        assert procname.BRAND not in argv0

    def test_an_unidentifiable_dylib_withholds_the_label_beside_a_target(self, tmp_path):
        """R-4 on the cross-tree axis, and the case the old order missed.

        The base tree's dylib is removed BEFORE any plant, so the interpreter
        stays executable and the PATH SHAPE is unchanged — only the companion
        becomes unidentifiable. The refusal must still be total, because a link
        handed out here would abort at load with nothing to see it.
        """
        target = _generation_venv(tmp_path / "gen")
        base = _generation_base(tmp_path / "gen")
        (base / "lib" / next(iter((base / "lib").iterdir())).name).unlink()

        argv0, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_DAEMONS_REFRESH, str(target)
        )

        assert (argv0, executable) == (str(target), None)
        assert not (target.parent / procname.BRAND).exists()

    def test_a_framework_target_is_refused(self, tmp_path):
        """A framework interpreter discards our name at its own stub re-exec.

        Skipped where this machine has no framework Python: the detection is
        path-based on the RESOLVED binary, so it cannot be faked by symlinking
        into a directory named ``Python.framework``.
        """
        candidates = [
            Path("/opt/homebrew/bin/python3.14"),
            Path("/opt/homebrew/bin/python3.13"),
            Path("/usr/local/bin/python3"),
        ]
        framework = next(
            (c for c in candidates if c.exists() and ".framework" in str(c.resolve())),
            None,
        )
        if framework is None:
            pytest.skip("no framework Python on this machine")
        root = tmp_path / "framework-venv"
        (root / "bin").mkdir(parents=True)
        (root / "pyvenv.cfg").write_text("home = /nowhere\n", encoding="utf-8")
        target = root / "bin" / "python3"
        os.symlink(framework, target)

        argv0, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_DAEMONS_REFRESH, str(target)
        )

        assert (argv0, executable) == (str(target), None)


class TestLaunchdJob:
    """``Program`` + role label: the shape three daemons could not be told apart in."""

    ROLES = (
        (
            "mobile",
            "local_operator.mobile.service",
            procname.LABEL_MOBILE,
            {"port": 4098},
            ["--port", "4098"],
        ),
        (
            "browser",
            "local_operator.browser_bridge.daemon",
            procname.LABEL_BROWSER,
            {"port": 4099},
            ["--port", "4099"],
        ),
        ("tunnel", "local_operator.tunnels.service", procname.LABEL_TUNNEL, {}, []),
        ("wakes", "local_operator.wakes.supervisor", procname.LABEL_WAKES, {}, []),
    )

    @pytest.mark.parametrize("role,module,template,fields,extra", ROLES)
    def test_program_key_carries_the_image_and_argv_the_label(
        self, role, module, template, fields, extra, branded
    ):
        label = procname.branded_argv0(template, **fields)
        job = procname.launchd_job(module, *extra, label=label)
        assert job["Program"] == str(branded), role
        assert job["ProgramArguments"] == [label, "-m", module, *extra], role

    def test_no_branded_image_means_the_pre_branding_plist(self, monkeypatch):
        """Rung 3 is byte-for-byte what these installers wrote before."""
        monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: None)
        job = procname.launchd_job(
            "local_operator.wakes.supervisor", label="Local Operator [wakes]"
        )
        assert job == {
            "ProgramArguments": [sys.executable, "-m", "local_operator.wakes.supervisor"]
        }
        assert "Program" not in job

    def test_the_label_is_optional(self, branded):
        job = procname.launchd_job("local_operator.wakes.supervisor")
        assert isinstance(job["ProgramArguments"], list)
        assert job["ProgramArguments"][0] == procname.BRAND

    @pytest.mark.parametrize("role,module,template,fields,extra", ROLES)
    def test_the_stable_shim_wins_over_the_per_venv_image(
        self, role, module, template, fields, extra, branded, tmp_path, monkeypatch
    ):
        """A unit names a path that survives a flip and a prune.

        The branded link lives inside ONE venv, with a libpython dylib pin
        beside it, and a supervised unit re-executes its image on every restart:
        naming that path is what made launchd respawn 113 processes into a tree
        the installer had already emptied on 2026-09-15. When the machine has the
        generation layout (``update.daemon_image()`` answers a path) the stable
        shim is the image and the LABEL stays where it was — the two axes are
        independent, which is the point of the branded shape.
        """
        shim = tmp_path / "bin" / "python3"
        monkeypatch.setattr(procname, "supervised_image", lambda: shim)
        label = procname.branded_argv0(template, **fields)
        job = procname.launchd_job(module, *extra, label=label)
        assert job == {
            "Program": str(shim),
            "ProgramArguments": [label, "-m", module, *extra],
        }, role
        assert procname.launchd_program(module, *extra) == [str(shim), "-m", module, *extra]

    def test_no_generation_layout_keeps_the_branded_shape(self, monkeypatch, branded):
        """A checkout or a pip install must render today's plist, unchanged.

        ``update.daemon_image()`` answers ``None`` there by design, and this
        pins that the fallback is the branded image rather than something new.
        """
        monkeypatch.setattr(procname, "supervised_image", lambda: None)
        job = procname.launchd_job(
            "local_operator.wakes.supervisor", label="Local Operator [wakes]"
        )
        assert job["Program"] == str(branded)
        assert procname.launchd_program("local_operator.wakes.supervisor")[0] == str(branded)


class TestResumeExecutableRegression:
    """``resume_executable`` must keep returning the ``lop`` LAUNCHER path.

    It reads ``sys.argv[0]`` precisely because ``sys.executable`` would restore
    a bare Python REPL instead of reopening the session. Branding rewrites
    argv[0] of CHILDREN, so this pins that the launcher's own resolution is
    unaffected — a regression here silently breaks every crash-restore and
    notification click, and would not show up as a failure anywhere else.
    """

    def test_resume_executable_is_argv0_not_the_interpreter(self, monkeypatch, tmp_path):
        from local_operator.multiplexer import broadcast

        launcher = tmp_path / "lop"
        launcher.write_text("#!/bin/sh\n")
        monkeypatch.setattr(sys, "argv", [str(launcher), "--resume", "abc"])
        resolved = broadcast.resume_executable()
        assert resolved == str(launcher.resolve())
        assert resolved != sys.executable

    def test_label_lives_in_orig_argv_not_sys_argv(self, branded):
        """The property that keeps `/reload` and crash-restore working.

        CPython sets ``sys.argv[0]`` from the SCRIPT path, so a branded process
        carries the label only in ``sys.orig_argv[0]``. If that ever changed,
        ``reexec.plan_argv`` would hand a label to ``os.execvpe``, which fails
        with ``FileNotFoundError`` and breaks every restore.
        """
        script = branded.parent.parent / "argv_report.py"
        script.write_text("import sys;print(sys.argv[0]);print(sys.orig_argv[0])\n")
        label = procname.branded_argv0(procname.LABEL_SERVE, port=1)
        result = subprocess.run(
            [label, str(script)],
            executable=str(branded),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        sys_argv0, orig_argv0 = result.stdout.split()[:2]
        assert sys_argv0 == str(script), "sys.argv[0] must stay the script path"
        assert orig_argv0 == "Local", "the label lives in orig_argv (first token)"

    def test_branded_argv0_does_not_leak_into_resume(self, monkeypatch):
        """A branded argv[0] is a LABEL, not a path — it must not be resolved.

        If a future change brands the main process's argv[0] with a label
        containing spaces, ``resume_executable`` must not hand that label back
        as an executable path.
        """
        from local_operator.multiplexer import broadcast

        monkeypatch.setattr(sys, "argv", [procname.branded_argv0(procname.LABEL_WAKES)])
        assert broadcast.resume_executable() != "Local Operator [wakes]"


class TestLinuxNameSet:
    """``prctl`` is capability-probed, not branched on a platform list."""

    def test_set_process_name_is_a_no_op_on_macos(self):
        # macOS has no PR_SET_NAME; the call must report False, never raise.
        assert procname.set_process_name("whatever") is False

    def test_set_process_name_never_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        # No ctypes/libc in this shape => False, not an exception.
        monkeypatch.setitem(sys.modules, "ctypes", None)
        assert procname.set_process_name() is False


class TestSpawnDetachedLabel:
    """``proc.spawn_detached(label=...)`` brands only Python spawns."""

    def test_label_is_ignored_for_a_non_interpreter_argv(self, monkeypatch, tmp_path):
        """A terminal emulator or ``open`` must keep its own image.

        Setting ``executable=`` there would run Python under that program's
        arguments — a spawn that silently does the wrong thing.
        """
        from local_operator import proc

        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["executable"] = kwargs.get("executable")

        monkeypatch.setattr(proc.subprocess, "Popen", _FakePopen)
        assert proc.spawn_detached(["/usr/bin/open", "-a", "Terminal"], label="x") is True
        assert captured["executable"] is None
        assert captured["argv"] == ["/usr/bin/open", "-a", "Terminal"]

    def test_spawn_without_label_is_unchanged(self, monkeypatch):
        from local_operator import proc

        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["executable"] = kwargs.get("executable")

        monkeypatch.setattr(proc.subprocess, "Popen", _FakePopen)
        assert proc.spawn_detached([sys.executable, "-c", "pass"]) is True
        assert captured["executable"] is None
        assert captured["argv"] == [sys.executable, "-c", "pass"]


class TestCrossTreeSpawnSites:
    """The two production sites that run ANOTHER tree's interpreter.

    A test of the helper alone would not stop a site going back to the shape
    that caused the incident — `spawn_identity`'s label kept while `executable`
    was overwritten with the other tree's bare interpreter. So each site is
    asserted on the argv/executable it REALLY builds, with the plant live on
    disk, and the same-tree branch is pinned next to it so the fix cannot turn
    into "always replant somewhere else".
    """

    def test_the_daemon_refresh_child_pairs_the_label_with_the_target_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact row SentinelOne killed: ``[daemons] refresh``.

        Its argv read ``Local Operator [daemons] refresh`` while its image was
        ``python3.14`` — the label without the image. ``update.current_interpreter``
        is faked to a throwaway generation venv, which is what the real one
        returns on a machine with the generation layout.
        """
        from local_operator import update

        target = _generation_venv(tmp_path / "gen")
        monkeypatch.setattr(update, "current_interpreter", lambda: target)

        plan = update._daemon_refresh_invocation()

        assert plan is not None
        argv, executable = plan
        assert executable == str(target.parent / procname.BRAND)
        assert argv == [
            "Local Operator [daemons] refresh",
            SAFE_PATH_FLAG,
            "-m",
            "local_operator.cli",
            "update",
            "--refresh-daemons",
            # The caller owns the mobile half (review round 2, MINOR-2): this child
            # must not bounce the phone relay its parent is about to bounce.
            "--services-only",
        ]
        assert os.access(str(executable), os.X_OK)

    def test_the_runtime_spawn_uses_the_target_trees_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_spawn_runtime`` on the engage-onto-the-next-generation path.

        Both halves are asserted together because they are two independent
        properties on one spawn: the label in ``argv[0]`` and the branded image
        beside the interpreter the child will ACTUALLY run — the current
        generation's, which is what makes a mixed-generation fleet converge.
        """
        from local_operator.session.runtime import launch as launch_module

        target = _generation_venv(tmp_path / "gen")
        monkeypatch.setattr(launch_module, "_spawn_interpreter", lambda: str(target))
        recorded: dict[str, Any] = {}

        class _Popen:
            returncode = None

            def poll(self):
                return None

            def kill(self):
                return None

        def fake_popen(argv, **kwargs: Any):
            recorded["argv"] = list(argv)
            recorded["executable"] = kwargs.get("executable")
            return _Popen()

        monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)
        process = launch_module._spawn_runtime("sess-xtree", str(tmp_path), defer_materialise=True)
        capture = getattr(process, "lop_capture_path", None)
        if capture is not None:
            capture.unlink(missing_ok=True)

        assert recorded["argv"][0] == "Local Operator [session] id=sess-xtr"
        assert recorded["argv"][1] == SAFE_PATH_FLAG
        assert recorded["executable"] == str(target.parent / procname.BRAND), (
            "the runtime must exec the branded link beside the generation's "
            "interpreter, not ours"
        )
        assert os.access(str(recorded["executable"]), os.X_OK)

    def test_a_same_tree_spawn_still_uses_this_processs_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other branch, pinned so the fix is not "plant somewhere else, always".

        ``_spawn_interpreter`` answers ``sys.executable`` whenever no generation
        layout is resolvable (a source checkout, a pip/pipx machine, an
        interrupted flip). Then the pair must come from ``spawn_identity`` —
        this process's own venv — exactly as it did before.
        """
        from local_operator.session.runtime import launch as launch_module

        monkeypatch.setattr(launch_module, "_spawn_interpreter", lambda: sys.executable)
        # A FAKE venv for "this process", never the worktree's real one. This
        # cell runs the REAL plant for ``sys.prefix``, so with the real prefix it
        # re-pointed the developer's ``<worktree>/.venv/bin/Local Operator`` into
        # pytest's tmp tree (observed twice) and left it dangling once that tree
        # was pruned. ``sys.prefix``/``base_prefix`` are patched the way the
        # ``branded`` fixture does it; the runtime root is the per-test one.
        prefix = _fake_venv(tmp_path / "own-venv")
        monkeypatch.setattr(sys, "prefix", str(prefix))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "own-base"))
        calls: list[str] = []
        real_pair = procname.spawn_identity
        real_cross = procname.spawn_identity_for_interpreter

        def spy_own(*args: Any, **kwargs: Any):
            calls.append("own")
            return real_pair(*args, **kwargs)

        def spy_cross(*args: Any, **kwargs: Any):
            calls.append("cross")
            return real_cross(*args, **kwargs)

        monkeypatch.setattr(procname, "spawn_identity", spy_own)
        monkeypatch.setattr(procname, "spawn_identity_for_interpreter", spy_cross)
        recorded: dict[str, Any] = {}

        class _Popen:
            returncode = None

            def poll(self):
                return None

            def kill(self):
                return None

        def fake_popen(argv, **kwargs: Any):
            recorded["executable"] = kwargs.get("executable")
            return _Popen()

        monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)
        process = launch_module._spawn_runtime("sess-own", str(tmp_path), defer_materialise=True)
        capture = getattr(process, "lop_capture_path", None)
        if capture is not None:
            capture.unlink(missing_ok=True)

        assert calls == ["own"], f"a same-tree spawn must not replant elsewhere: {calls}"
        assert recorded["executable"] in (None, str(procname.ensure_branded_interpreter()))


class TestStableRuntimeImage:
    """One image per Python build, and every venv links to it.

    The properties that make macOS's privacy rows stop multiplying: the image
    is a clone (own inode), its path is a FUNCTION of the interpreter build,
    and every venv — generation or worktree — resolves through the same one.
    """

    def _copied_tree(self, tmp_path: Path) -> Path:
        """A throwaway interpreter tree with a real ``lib``, safe to mutate.

        The operator's uv interpreter is NEVER touched: bumping its mtime or
        unlinking its dylib would re-key every venv on the machine (and is the
        2026-09-19 blast radius this layout exists to remove).
        """
        real = Path(os.path.realpath(sys.executable))
        base = tmp_path / "base"
        (base / "bin").mkdir(parents=True)
        (base / "lib").mkdir(parents=True)
        copied = base / "bin" / real.name
        shutil.copy2(real, copied)
        for entry in (real.parent.parent / "lib").iterdir():
            if entry.name.startswith("python") or entry.name.startswith("libpython"):
                os.symlink(entry, base / "lib" / entry.name)
        return copied

    def test_the_plant_publishes_a_clone_beside_a_stdlib_link(self, branded):
        image = Path(os.readlink(branded))
        real = procname.real_interpreter()
        assert real is not None
        real_stat = Path(os.path.realpath(real)).stat()
        image_stat = image.stat()

        assert image_stat.st_nlink == 1, "the image must not share uv's inode"
        assert image_stat.st_ino != real_stat.st_ino
        assert (image_stat.st_size, image_stat.st_mtime_ns) == (
            real_stat.st_size,
            real_stat.st_mtime_ns,
        ), "a clone keeps the bytes AND the timestamps"
        assert (image.parent.parent / "lib").is_symlink()
        assert (image.parent.parent / "lib").resolve() == (real.parent.parent / "lib").resolve()
        assert (image.parent.parent / "include").is_symlink()
        assert os.readlink(branded) == str(image)
        assert branded.parent.name == "bin" and image.name == procname.BRAND

    def test_executing_the_venv_link_names_this_process_and_keeps_the_venv(self, tmp_path):
        """The triple TCC, Activity Monitor and Python each read differently.

        Twenty launches, not one: the warm-dyld trap in the module docstring is
        exactly a first-run success followed by 100% failure, so a single smoke
        run proves nothing about this shape.
        """
        target = _runnable_venv(tmp_path)
        venv = target.parent.parent
        _, link = procname.spawn_identity_for_interpreter(procname.LABEL_SERVE, target, port=1)
        assert link is not None, "the runtime rung must be available here"
        link = Path(link)
        assert link.is_symlink()
        probe = (
            "import ctypes, os, sys;"
            "l=ctypes.CDLL('/usr/lib/libSystem.dylib');"
            "b=ctypes.create_string_buffer(64);"
            "l.proc_name(ctypes.c_int(os.getpid()),b,ctypes.c_uint(64));"
            "print(b.value.decode());print(sys.prefix)"
        )
        for launch in range(20):
            result = subprocess.run(
                [procname.BRAND, SAFE_PATH_FLAG, "-c", probe],
                executable=str(link),
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, (launch, result.stderr)
            name, prefix = result.stdout.strip().splitlines()[:2]
            assert name == procname.BRAND, (launch, name)
            assert os.path.realpath(prefix) == os.path.realpath(venv), (launch, prefix)

    def test_two_venvs_share_one_image(self, branded, tmp_path):
        """A worktree venv and a generation venv are the SAME privacy client."""
        real = procname.real_interpreter()
        assert real is not None
        second = _fake_venv(tmp_path / "second")
        (second / "pyvenv.cfg").write_text(f"home = {real.parent}\n", encoding="utf-8")
        (second / "bin" / "python3").symlink_to(real)

        argv0, executable = procname.spawn_identity_for_interpreter(
            procname.LABEL_SERVE, second / "bin" / "python3", port=1
        )

        assert executable is not None
        assert os.readlink(executable) == os.readlink(branded)
        root = procname.runtime_root()
        assert root is not None
        assert len([p for p in root.iterdir() if p.is_dir()]) == 1

    def test_a_new_interpreter_build_publishes_a_new_image(self, tmp_path):
        """A replaced interpreter is a NEW key; the old image is left alone."""
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        first = procname._plant_runtime(real, root)
        assert first is not None
        before = {p.name: p.stat().st_mtime_ns for p in root.iterdir() if p.is_dir()}

        bumped = real.stat().st_mtime_ns + 1_000_000_000
        os.utime(real, ns=(bumped, bumped))
        second = procname._plant_runtime(real, root)

        assert second is not None and second != first
        assert first.parent.parent.is_dir(), "a published image is never collected"
        after = {p.name: p.stat().st_mtime_ns for p in root.iterdir() if p.is_dir()}
        assert set(after) == set(before) | {second.parent.parent.name}
        for name, mtime in before.items():
            assert after[name] == mtime, "the published directory is immutable"

    def test_the_daemon_shim_runs_the_branded_image_and_falls_back_when_broken(
        self, tmp_path, branded
    ):
        """The shim asks the stdlib question at the IMAGE's directory, then uses it.

        BOTH halves are named, because the first version of this cell could not
        see the bug it was written for: its "good" tree asserted only that the
        interpreter printed ``ok``, which the FALLBACK interpreter prints too,
        and the glob probed ``<key>/bin/lib`` where the plant publishes
        ``<key>/lib``. The positive control now asks the KERNEL which image ran
        (``proc_name``), so a fallback cannot satisfy it, and the negative
        replaces the sibling interpreter with a script that says so.
        """
        from local_operator import update as update_mod

        def shim_tree(image: Path) -> Path:
            stable = tmp_path / "stable"
            (stable / "bin").mkdir(parents=True, exist_ok=True)
            written = stable / "bin" / "python3"
            written.write_text(update_mod._DAEMON_SHIM, encoding="utf-8")
            written.chmod(0o755)
            gen = stable / "generations" / "g1"
            bin_dir = gen / "tools" / "local-operator" / "bin"
            bin_dir.mkdir(parents=True)
            os.symlink("generations/g1", stable / "current")
            (bin_dir / "Local Operator").symlink_to(image)
            (bin_dir / "python3").symlink_to(Path(os.path.realpath(sys.executable)))
            return written

        probe = (
            "import ctypes, os;"
            "l=ctypes.CDLL('/usr/lib/libSystem.dylib');"
            "b=ctypes.create_string_buffer(64);"
            "l.proc_name(ctypes.c_int(os.getpid()),b,ctypes.c_uint(64));"
            "print(b.value.decode())"
        )
        image = Path(os.readlink(branded))
        good = shim_tree(image)
        ran = subprocess.run([str(good), "-c", probe], capture_output=True, text=True, timeout=60)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.strip() == procname.BRAND, (
            "the shim must exec the branded image, not the fallback interpreter "
            f"(kernel reported {ran.stdout.strip()!r})"
        )

        # Same layout and same name, but the image's stdlib no longer resolves:
        # the shim must exec the sibling interpreter instead of that image, and
        # the fallback is a script so a pass here cannot be the image running.
        bin_dir = tmp_path / "stable" / "generations" / "g1" / "tools" / "local-operator" / "bin"
        (image.parent.parent / "lib").unlink()
        os.symlink("/nonexistent/lib", image.parent.parent / "lib")
        (bin_dir / "python3").unlink()
        (bin_dir / "python3").write_text("#!/bin/sh\necho fallback\n")
        (bin_dir / "python3").chmod(0o755)

        ran = subprocess.run(
            [str(good), "-c", "print('should-not-run')"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.strip() == "fallback", ran.stdout

    @staticmethod
    def _sized_shim_tree(tmp_path: Path, image: Path, size: int) -> Path:
        """A shim tree whose fallback ``python3`` is a script of exactly ``size`` bytes.

        The shim's integrity probe compares the image's size with the sibling
        interpreter's, so a fallback of the image's size isolates the one
        variable the cell is about: the IMAGE's bytes. The script says
        ``fallback`` and exits before reading its padding.
        """
        from local_operator import update as update_mod

        stable = tmp_path / "stable"
        (stable / "bin").mkdir(parents=True)
        shim = stable / "bin" / "python3"
        shim.write_text(update_mod._DAEMON_SHIM, encoding="utf-8")
        shim.chmod(0o755)
        bin_dir = stable / "generations" / "g1" / "tools" / "local-operator" / "bin"
        bin_dir.mkdir(parents=True)
        os.symlink("generations/g1", stable / "current")
        (bin_dir / "Local Operator").symlink_to(image)
        script = b"#!/bin/sh\necho fallback\nexit 0\n"
        (bin_dir / "python3").write_bytes(script + b"#" * (size - len(script)))
        (bin_dir / "python3").chmod(0o755)
        return shim

    def test_the_daemon_shim_execs_a_healthy_image_of_the_fallbacks_size(self, tmp_path):
        """Positive control for the shim's size probe: equal sizes run the IMAGE."""
        real = self._copied_tree(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")
        assert image is not None
        shim = self._sized_shim_tree(tmp_path, image, image.stat().st_size)

        ran = subprocess.run(
            [str(shim), "-c", "print('image-ran')"], capture_output=True, text=True, timeout=60
        )

        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.strip() == "image-ran", ran.stdout

    def test_the_daemon_shim_falls_back_from_a_truncated_image(self, tmp_path):
        """Item 3: a truncated image passes ``-x`` and the stdlib probe, then dies.

        Without the size probe the shim exec'd it and every supervised daemon
        failed with ``Exec format error``; with it the sibling interpreter runs.
        """
        real = self._copied_tree(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")
        assert image is not None
        size = image.stat().st_size
        with open(image, "r+b") as handle:
            handle.truncate(12)
        shim = self._sized_shim_tree(tmp_path, image, size)

        ran = subprocess.run(
            [str(shim), "-c", "print('image-ran')"], capture_output=True, text=True, timeout=60
        )

        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.strip() == "fallback", ran.stdout

    def test_concurrent_planters_publish_one_image_and_leak_no_temps(self, tmp_path):
        import concurrent.futures

        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: procname._plant_runtime(real, root), range(8)))

        assert all(result is not None for result in results)
        assert len({str(result) for result in results}) == 1
        assert len([p for p in root.iterdir() if p.is_dir()]) == 1
        assert not list(root.glob("*.tmp.*")), "every planter must consume its own temp"
        image = results[0]
        assert image is not None and procname._healthy_image(image)

    def test_concurrent_planters_repair_a_corrupt_image_without_a_gap(self, tmp_path):
        """Review round 2, item 4: the repair used to delete the key IN PLACE.

        ``rmtree(<key>)`` then ``rename(staging)`` left the key missing for as
        long as the delete ran, and planters that looked in that window answered
        ``None`` — measured 139 of 240 — each dropping a venv to the hardlink
        rung. Moving the dead tree aside is atomic; a loser of the final rename
        re-checks once. Several rounds, because one round can miss the window.
        """
        import concurrent.futures

        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        for _ in range(5):
            image = procname._plant_runtime(real, root)
            assert image is not None
            with open(image, "r+b") as handle:
                handle.truncate(12)
            assert not procname._healthy_image(image, real), "precondition: corrupt start"

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: procname._plant_runtime(real, root), range(8)))

            assert all(result is not None for result in results), results
            assert {str(result) for result in results} == {str(image)}
            assert [p.name for p in root.iterdir() if p.is_dir()] == [image.parent.parent.name]
            assert procname._healthy_image(image, real)
            assert not list(root.glob("*.tmp.*")), "a staging tree was leaked"
            assert not list(root.glob("*.dead.*")), "a moved-aside tree was leaked"

    def test_a_dead_planters_aside_tree_is_swept_and_a_live_ones_kept(self, tmp_path):
        """``<key>.dead.<pid>.<thread>`` gets the staging tree's liveness rule."""
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        root.mkdir()
        dead = root / "cpython-key.dead.999999.1"
        live = root / f"cpython-key.dead.{os.getppid()}.1"
        for entry in (dead, live):
            (entry / "bin").mkdir(parents=True)

        assert procname._plant_runtime(real, root) is not None

        assert not dead.exists(), "a dead planter's aside tree must be reclaimed"
        assert live.exists(), "a live planter's aside tree must be left alone"

    def test_an_unwritable_runtime_root_is_a_rung_not_an_error(self, tmp_path):
        """A sandboxed child cannot publish; it must say so, not raise."""
        real = self._copied_tree(tmp_path)
        root = tmp_path / "readonly"
        root.mkdir()
        root.chmod(0o500)
        try:
            assert procname._plant_runtime(real, root) is None
        finally:
            root.chmod(0o700)
        assert not list(root.glob("*.tmp.*")), "a failed plant cleans up after itself"

    def test_a_healthy_runtime_link_is_never_replaced_by_the_fallback(self, branded, monkeypatch):
        """The sandboxed-child case: cannot plant, must not downgrade either.

        A read-only runtime root makes the runtime rung unavailable, and the
        hardlink rung below it must then leave a HEALTHY runtime symlink alone.
        Replacing it would give the child a second path (a second privacy
        prompt) and undo what its parent planted.

        THE STEADY STATE IS NOT THE STATE THIS GUARD LIVES IN, so the cell has
        to be driven past it: the previous version of this test passed with the
        guard replaced by ``if False:`` because a healthy link is already
        returned by :func:`_needs_replant` one branch above. Here the build key
        MOVED (``runtime_image`` names a key nothing published, which is what
        ``uv tool install --force`` does to an installed tree) and the plant is
        refused, which is the pair of facts that reaches the guard.
        """
        image = Path(os.readlink(branded))
        moved = image.parent.parent.parent / "a-key-nothing-published" / "bin" / procname.BRAND
        monkeypatch.setattr(procname, "runtime_image", lambda real: moved)
        monkeypatch.setattr(procname, "_plant_runtime", lambda real, root: None)

        assert procname.ensure_branded_interpreter() == branded

        assert branded.is_symlink(), "a healthy runtime link must survive rung 2"
        assert os.readlink(branded) == str(image)

    def test_a_truncated_clone_is_repaired_rather_than_handed_out(self, tmp_path):
        """M2: ``X_OK`` plus a resolvable stdlib passed a 12-byte image.

        A truncated clone is executable and its stdlib still resolves, so the
        old health question answered yes and the kernel answered ``[Errno 8]
        Exec format error``. One image is shared by every venv of the build, so
        that state disables the brand for all of them until someone deletes the
        directory by hand; the key already encodes the source's size, and the
        repair republishes it under the same key.
        """
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        image = procname._plant_runtime(real, root)
        assert image is not None
        link = tmp_path / "venv" / "bin" / procname.BRAND
        link.parent.mkdir(parents=True)
        os.symlink(image, link)

        with open(image, "r+b") as handle:
            handle.truncate(12)

        assert not procname._healthy_image(image, real)
        assert procname._needs_replant(link, real, image), "a truncated image is stale"
        assert procname._executable_image(link, real) is None, "never hand out a dead image"

        repaired = procname._plant_runtime(real, root)

        assert repaired == image, "repaired in place, under the same key"
        assert image.stat().st_size == real.stat().st_size
        assert procname._healthy_image(image, real)
        assert procname._executable_image(link, real) == link

    def test_an_image_overwritten_in_place_at_the_same_size_is_repaired(self, tmp_path):
        """Review round 2, item 1: size alone passed a zeroed header.

        ``open(img, 'r+b').write(b'\\0' * 64)`` keeps the size and the stdlib
        link, and every spawn then got ``Exec format error``. The write stamps a
        new mtime, and a clone keeps its source's, so the mtime catches it.
        """
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        image = procname._plant_runtime(real, root)
        assert image is not None
        link = tmp_path / "venv" / "bin" / procname.BRAND
        link.parent.mkdir(parents=True)
        os.symlink(image, link)

        with open(image, "r+b") as handle:
            handle.write(b"\0" * 64)

        assert image.stat().st_size == real.stat().st_size, "precondition: same size"
        assert not procname._healthy_image(image, real)
        assert procname._needs_replant(link, real, image)
        assert procname._executable_image(link, real) is None
        assert procname._plant_runtime(real, root) == image
        assert procname._healthy_image(image, real), "the repair republishes the key"

    def test_a_script_restored_to_the_images_size_and_mtime_is_not_an_image(self, tmp_path):
        """The magic check: size and mtime put back cannot vouch for the bytes."""
        real = self._copied_tree(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")
        assert image is not None
        source = real.stat()
        script = b"#!/bin/sh\nexit 0\n"
        image.write_bytes(script + b"#" * (source.st_size - len(script)))
        os.utime(image, ns=(source.st_atime_ns, source.st_mtime_ns))

        assert (image.stat().st_size, image.stat().st_mtime_ns) == (
            source.st_size,
            source.st_mtime_ns,
        ), "precondition: size and mtime both match the source"
        assert not procname._healthy_image(image, real)
        assert not procname._healthy_image(image), "the magic is checked without a source too"
        assert procname._executable_image(image, real) is None

    def test_a_same_size_image_under_another_key_is_replanted(self, branded, monkeypatch):
        """Mutation (c): the "link names THIS exact image" check in ``_needs_replant``.

        An mtime-only reinstall keeps the size, so the old image is a healthy,
        same-size file under a DIFFERENT key. Only the path comparison says the
        link is stale; the health check alone would keep it.
        """
        real = procname.real_interpreter()
        assert real is not None
        old = Path(os.readlink(branded))
        moved = old.parent.parent.parent / "cpython-moved-key" / "bin" / procname.BRAND
        assert procname._healthy_image(old, real), "precondition: the old image is healthy"
        assert not procname._needs_replant(branded, real, old), "precondition: steady state"

        assert procname._needs_replant(branded, real, moved)

    def test_a_link_straight_at_the_interpreter_is_not_the_runtime_link(self, tmp_path):
        """Mutation (d): ``_is_runtime_link``'s "target under the runtime root" check.

        A symlink straight at the interpreter executes and its stdlib resolves,
        so every health question passes — and it reports ``python3.x``, one
        privacy client per interpreter rather than the image. It must not be
        kept as if it were the runtime link.
        """
        real = self._copied_tree(tmp_path)
        link = tmp_path / "venv" / "bin" / procname.BRAND
        link.parent.mkdir(parents=True)
        os.symlink(real, link)
        assert procname._healthy_image(link, real), "precondition: it is a healthy image"

        assert not procname._is_runtime_link(link, real)

    def test_a_failed_build_never_publishes_a_half_built_directory(self, tmp_path, monkeypatch):
        """m1: the ``except OSError`` caught a failing clone and renamed the staging tree.

        A build that dies (ENOSPC here) left ``<root>/<key>/bin`` behind: a key a
        reader can see, and a next planter sent down the corrupt-repair path for
        a directory that never held an image. A failed build is not a publish.
        """
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"

        def out_of_space(source: Path, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"half a clone")
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(procname, "_clone_or_copy", out_of_space)

        assert procname._plant_runtime(real, root) is None
        assert not [
            entry for entry in root.iterdir() if entry.is_dir()
        ], "a failed build published a directory"
        assert not list(root.glob("*.tmp.*")), "the staging tree was left behind"

    def test_a_venv_built_on_an_image_maps_back_to_the_source(self, tmp_path, monkeypatch):
        """m2: ``<venv>/bin/Local Operator -m venv X`` makes the CLONE the base.

        That venv's ``pyvenv.cfg`` names the image directory, so resolving it
        answered the clone and the next plant published a clone OF a clone under
        a second key — a second macOS privacy row for one Python build. The
        ``source`` marker written at publish is what maps it back.
        """
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(root))
        image = procname._plant_runtime(real, root)
        assert image is not None
        marker = image.parent.parent / procname._SOURCE_MARKER
        assert marker.read_text(encoding="utf-8").strip() == os.path.realpath(real)

        rebuilt = tmp_path / "rebuilt"
        (rebuilt / "bin").mkdir(parents=True)
        (rebuilt / "pyvenv.cfg").write_text(
            f"home = {image.parent}\nexecutable = {image}\n", encoding="utf-8"
        )
        (rebuilt / "bin" / "python").symlink_to(image)
        monkeypatch.setattr(sys, "prefix", str(rebuilt))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
        monkeypatch.setattr(sys, "executable", str(rebuilt / "bin" / "python"))

        resolved = procname.real_interpreter()

        assert resolved is not None
        assert resolved == Path(
            os.path.realpath(real)
        ), "a venv built on an image must answer the build it was cloned from"
        assert procname.runtime_image(resolved) == image, "no clone of a clone"

    def test_a_venv_on_an_unmapped_image_answers_no_build(self, tmp_path, monkeypatch):
        """Item 5: no marker, so the clone is the only answer — and it is refused.

        Same venv shape as the cell above (``Local Operator -m venv``), but the
        image's ``source`` marker is gone. Answering the clone would make the
        next plant publish a clone of it under a second key; ``None`` means no
        plant for this venv and nothing else.
        """
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(root))
        image = procname._plant_runtime(real, root)
        assert image is not None
        (image.parent.parent / procname._SOURCE_MARKER).unlink()
        rebuilt = tmp_path / "rebuilt"
        (rebuilt / "bin").mkdir(parents=True)
        (rebuilt / "pyvenv.cfg").write_text(
            f"home = {image.parent}\nexecutable = {image}\n", encoding="utf-8"
        )
        (rebuilt / "bin" / "python").symlink_to(image)
        monkeypatch.setattr(sys, "prefix", str(rebuilt))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
        monkeypatch.setattr(sys, "executable", str(rebuilt / "bin" / "python"))

        assert procname.real_interpreter() is None
        assert procname._real_interpreter_for(rebuilt / "bin" / "python") is None
        assert [p.name for p in root.iterdir()] == [image.parent.parent.name], "no second key"

    def test_real_interpreter_reads_pyvenv_cfg_not_the_executable(self, tmp_path, monkeypatch):
        """Mutation (b): skipping ``pyvenv.cfg`` must answer the wrong build.

        No marker anywhere. The venv's ``pyvenv.cfg`` names one interpreter
        and ``sys.executable`` resolves to a DIFFERENT one, so only reading the
        config gives the build the venv was created from.
        """
        real = self._copied_tree(tmp_path)
        other = tmp_path / "other" / "bin" / real.name
        other.parent.mkdir(parents=True)
        shutil.copy2(real, other)
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(f"home = {real.parent}\n", encoding="utf-8")
        (venv / "bin" / real.name).symlink_to(other)
        monkeypatch.setattr(sys, "prefix", str(venv))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
        monkeypatch.setattr(sys, "executable", str(venv / "bin" / real.name))

        assert procname.real_interpreter() == Path(os.path.realpath(real))

    def test_a_marker_naming_a_python_inside_the_runtime_tree_is_ignored(
        self, tmp_path, monkeypatch
    ):
        """Mutation (a): the "target outside the publish tree" guard.

        The cell below names the image itself, which the ``python*`` name check
        refuses anyway, so it passed with this guard removed. A ``python3.12``
        UNDER the runtime root passes every other check, and is still a clone.
        """
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(root))
        image = procname._plant_runtime(real, root)
        assert image is not None
        inside = root / "stray" / "bin" / "python3.12"
        inside.parent.mkdir(parents=True)
        shutil.copy2(real, inside)
        (image.parent.parent / procname._SOURCE_MARKER).write_text(f"{inside}\n", encoding="utf-8")

        assert procname._source_for_image(image) is None

    def test_a_marker_naming_a_foreign_binary_is_ignored(self, tmp_path, monkeypatch):
        """m-b: the marker was read unvalidated, so any binary could be the answer.

        Probe from the review: a marker rewritten to ``/bin/ls`` made
        ``_source_for_image`` RETURN ``/bin/ls``, and the next plant would have
        published a clone of it. The marker is a statement about OUR layout, so
        it is honoured only for a published image, and only when its target is
        an executable regular file OUTSIDE the publish tree — otherwise the
        answer is another clone and the loop this exists to stop comes back.
        """
        real = self._copied_tree(tmp_path)
        root = tmp_path / "runtime"
        monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(root))
        image = procname._plant_runtime(real, root)
        assert image is not None
        marker = image.parent.parent / procname._SOURCE_MARKER

        marker.write_text("/bin/ls\n", encoding="utf-8")
        assert procname._source_for_image(image) is None, "a foreign binary is not a source"

        marker.write_text(f"{os.path.realpath(image)}\n", encoding="utf-8")
        assert procname._source_for_image(image) is None, "the target must be outside the tree"

        marker.write_text(f"{os.path.realpath(real)}\n", encoding="utf-8")
        assert procname._source_for_image(image) == Path(
            os.path.realpath(real)
        ), "the same image with its real marker must still map back"

    def test_a_marker_beside_an_unpublished_interpreter_is_ignored(self, tmp_path, monkeypatch):
        """The marker is read at ``<path>/../../source``, which exists in other trees.

        A pyenv, framework or checkout tree is not a published image, so a file
        called ``source`` one level up there is not ours to read — a venv built
        on such a tree used to answer whatever it named. A published image is
        ``<runtime_root>/<key>/bin/<BRAND>``; the guard is that layout, not the
        file name, which the second half of the cell proves.
        """
        monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(tmp_path / "runtime"))
        tree = tmp_path / "pyenv" / "3.12"
        (tree / "bin").mkdir(parents=True)
        interpreter = tree / "bin" / "python3.12"
        interpreter.write_bytes(b"#!/bin/sh\nexit 0\n")
        interpreter.chmod(0o755)
        (tree / procname._SOURCE_MARKER).write_text(
            f"{os.path.realpath(interpreter)}\n", encoding="utf-8"
        )

        assert procname._source_for_image(interpreter) is None

        published_dir = tmp_path / "runtime" / "cpython-3.12-key" / "bin"
        published_dir.mkdir(parents=True)
        published = published_dir / procname.BRAND
        published.write_bytes(b"#!/bin/sh\nexit 0\n")
        published.chmod(0o755)
        (published_dir.parent / procname._SOURCE_MARKER).write_text(
            f"{os.path.realpath(interpreter)}\n", encoding="utf-8"
        )

        assert procname._source_for_image(published) == Path(os.path.realpath(interpreter))

    def test_a_venv_resident_symlink_asks_the_stdlib_question(self, tmp_path):
        """Q2: the stdlib gate was skipped for exactly the shape that needs it.

        ``<venv>/bin/Local Operator`` is a symlink to the runtime image, and the
        skip was keyed on ``pyvenv.cfg`` sitting beside the LINK — which is true
        for every venv, so the gate only ever ran for the image path. CPython
        takes the stdlib from the directory of the image it ends up running, so
        the question belongs to the SYMLINK. An in-venv HARDLINK stays exempt: it
        IS the venv's interpreter, and CPython reads that venv's own config.
        """
        real = self._copied_tree(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")
        assert image is not None
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(f"home = {image.parent}\n", encoding="utf-8")
        link = venv / "bin" / procname.BRAND
        os.symlink(image, link)
        (image.parent.parent / "lib").unlink()
        os.symlink("/nonexistent/lib", image.parent.parent / "lib")

        assert (link.parent.parent / "pyvenv.cfg").is_file(), "precondition: this venv shape"
        assert procname._executable_image(image) is None
        assert (
            procname._executable_image(link, real) is None
        ), "a venv link to an image with no stdlib must not reach execve"

    def test_runtime_root_without_an_absolute_home_answers_none(self, monkeypatch, tmp_path):
        """m3: ``Path("~/")`` is never expanded, so the plant created ``./~``.

        Reproduced by removing the passwd answer: ``mkdir(parents=True)`` made a
        literal ``~`` directory in the CURRENT working directory, inside whatever
        checkout the process happened to run from. With nowhere legitimate to
        publish, the root says so and the caller takes rung 2.
        """
        import pwd as pwd_module
        from types import SimpleNamespace

        monkeypatch.delenv("LOCAL_OPERATOR_RUNTIME_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)

        def no_entry(uid: int) -> SimpleNamespace:
            raise KeyError(uid)

        monkeypatch.setattr(pwd_module, "getpwuid", no_entry)
        assert procname.runtime_root() is None

        monkeypatch.setattr(
            pwd_module, "getpwuid", lambda uid: SimpleNamespace(pw_dir="relative/home")
        )
        assert procname.runtime_root() is None
        assert not (tmp_path / "~").exists(), "nothing may be created in the working directory"
        assert procname._plant_runtime(Path(sys.executable), None) is None

    def test_the_runtime_root_does_not_follow_home(self, monkeypatch):
        """SDK children get ``HOME`` rewritten; the privacy client must not move.

        ``agent/sdk.py`` sets ``HOME`` to an agent home for SDK children, so a
        root derived from the environment would plant a second image — a second
        privacy prompt — for every such child. The passwd entry is the answer
        that does not move, and this pins that it is the one consulted.
        """
        import pwd as pwd_module
        from types import SimpleNamespace

        monkeypatch.delenv("LOCAL_OPERATOR_RUNTIME_ROOT", raising=False)
        monkeypatch.setenv("HOME", "/nonexistent/agent-home")
        monkeypatch.setattr(
            pwd_module, "getpwuid", lambda uid: SimpleNamespace(pw_dir="/Users/real")
        )
        assert procname.runtime_root() == Path("/Users/real/.local/share/lop/runtime")

    def test_a_legacy_hardlink_is_migrated_to_the_symlink(self, branded, tmp_path):
        real = Path(os.path.realpath(sys.executable))
        branded.unlink()
        os.link(real, branded)
        assert not branded.is_symlink()

        assert procname.ensure_branded_interpreter() == branded
        assert branded.is_symlink()
        real = procname.real_interpreter()
        assert real is not None
        assert os.readlink(branded) == str(procname.runtime_image(real))

    def test_real_interpreter_looks_through_the_venv_symlink(self, tmp_path):
        """Running through the link, ``realpath(sys.executable)`` is the CLONE.

        So the base interpreter must come from the venv's ``pyvenv.cfg``
        instead — otherwise every plant after the first would clone a clone.
        """
        target = _runnable_venv(tmp_path)
        _, link = procname.spawn_identity_for_interpreter(procname.LABEL_SERVE, target, port=1)
        assert link is not None
        script = tmp_path / "which.py"
        script.write_text(
            "import os, sys;"
            "from local_operator import procname;"
            "print(os.path.realpath(sys.executable));"
            "print(procname.real_interpreter())",
            encoding="utf-8",
        )
        result = subprocess.run(
            [procname.BRAND, str(script)],
            executable=str(link),
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])),
        )
        assert result.returncode == 0, result.stderr
        running_as, base = result.stdout.strip().splitlines()[:2]
        assert Path(running_as) == Path(os.readlink(link)), "precondition: the clone"
        assert Path(base) == Path(os.path.realpath(target))


def test_branded_image_actually_renames_the_process(branded):
    """The end-to-end property, run for real: exec through the link and ask the kernel.

    This is the only assertion that proves the FEATURE rather than its
    plumbing. It also covers the warm-dyld-cache trap in the one direction a
    test can: a run that aborts (``Abort trap: 6``) because the libpython
    symlink is missing fails here loudly.
    """
    probe = (
        "import ctypes,os;"
        "l=ctypes.CDLL('/usr/lib/libSystem.dylib');"
        "b=ctypes.create_string_buffer(64);"
        "l.proc_name(ctypes.c_int(os.getpid()),b,ctypes.c_uint(64));"
        "print(b.value.decode())"
    )
    result = subprocess.run(
        [procname.branded_argv0(procname.LABEL_WAKES), "-c", probe],
        executable=str(branded),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"branded image failed to launch: {result.stderr}"
    assert result.stdout.strip() == procname.BRAND


def test_branded_image_and_import_isolation_hold_on_the_same_spawn(branded, tmp_path):
    """BOTH properties on ONE process — the pair a clean merge can silently halve.

    Branding and import isolation were developed in parallel and land on the
    SAME spawn calls with disjoint arguments: branding replaces ``argv[0]`` and
    passes ``executable=``, isolation inserts ``SAFE_PATH_FLAG`` before ``-m``.
    Git merges that without a conflict, and neither side's own tests can observe
    the other being dropped — a child that keeps isolation but loses
    ``executable=`` imports the right code while going back to an anonymous
    ``python3.x`` row, and the reverse keeps the name while keeping the
    version-skew bug that left live runtimes reporting a superseded build.

    So this asserts the CONJUNCTION from a real spawn, with a decoy checkout as
    the working directory (the exact condition that produced the skew):

    * the kernel reports ``proc_name == BRAND`` — Activity Monitor's axis, and
    * the child imports ``local_operator`` from the install, not from the cwd.
    """
    decoy = tmp_path / "decoy-checkout"
    (decoy / "local_operator").mkdir(parents=True)
    (decoy / "local_operator" / "__init__.py").write_text("", encoding="utf-8")
    decoy_module = (decoy / "local_operator" / "__init__.py").resolve()

    probe = (
        "import ctypes,os,local_operator;"
        "l=ctypes.CDLL('/usr/lib/libSystem.dylib');"
        "b=ctypes.create_string_buffer(64);"
        "l.proc_name(ctypes.c_int(os.getpid()),b,ctypes.c_uint(64));"
        "print(b.value.decode());"
        "print(local_operator.__file__)"
    )
    # The branded link is planted into a throwaway prefix that has no
    # site-packages, so the "install" this child must prefer is supplied on
    # PYTHONPATH. That is deliberate and does not weaken the test: `-P` strips
    # the implicit sys.path[0] (the cwd) and leaves PYTHONPATH alone, so the
    # decoy and the install compete exactly as they do in production.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        # Exactly the argv shape the spawn sites build: branded argv[0], then
        # the isolation flag, then the request.
        [procname.branded_argv0(procname.LABEL_WAKES), SAFE_PATH_FLAG, "-c", probe],
        executable=str(branded),
        cwd=str(decoy),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"spawn failed: {result.stderr}"
    name, module = result.stdout.strip().splitlines()[:2]
    assert name == procname.BRAND
    assert Path(module).resolve() != decoy_module


# --- the CLI's own gate: WHO re-execs ---------------------------------------


def test_the_cli_brand_gate_skips_the_short_lived_exec_paths(monkeypatch) -> None:
    """The gate's contract, pinned at the seam the re-exec is called from.

    ``lop exec --background`` exits in seconds once its detached worker is up,
    and ``lop exec --status`` exits on the first ledger read — neither lives
    long enough for the branded Activity Monitor row to be the point, and the
    WORKER (born branded via ``spawn_identity``) is the process whose row
    matters. Foreground ``exec`` and every interactive/long-lived subcommand
    keep the re-exec. Measured when the skip was added (2026-10-01): the
    re-exec costs +0.2-0.4 s per invocation, paid once per child in a bulk
    resume.
    """
    import argparse

    from local_operator import cli as cli_module

    recorded: list[object] = []

    def record(label=None):
        recorded.append(label)

    monkeypatch.setattr(cli_module.procname, "reexec_branded", record)

    def run(**args):
        del recorded[:]
        cli_module._maybe_brand_process(argparse.Namespace(**args))
        return list(recorded)

    # The two short-lived exec shapes: no re-exec.
    assert run(subcommand="exec", background=True) == []
    assert run(subcommand="exec", background=True, status=None) == []
    assert run(subcommand="exec", status="abc123") == []
    # Foreground exec (the run lives in THIS process): re-exec, bare brand.
    assert run(subcommand="exec", background=False) == [None]
    # Long-lived subcommands and the interactive TUI keep branding.
    assert len(run(subcommand="serve", port=1)) == 1
    assert len(run(subcommand=None)) == 1
    # Subcommands outside the brand set are untouched, background or not.
    assert run(subcommand="sessions", background=False) == []

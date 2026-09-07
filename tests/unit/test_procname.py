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
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator import procname

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the branded-image mechanism is macOS-specific; Linux uses prctl",
)


def _fake_venv(tmp_path: Path) -> Path:
    """A venv-shaped tree whose 'interpreter' is a hardlink of the real one.

    Real, because the thing under test IS filesystem behaviour: inode identity,
    link counts and symlink resolution cannot be faked with a mock and still
    prove anything.
    """
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lib").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def branded(tmp_path, monkeypatch):
    """Point ``procname`` at a throwaway prefix and plant into it.

    ``sys.prefix``/``sys.base_prefix`` are patched so the module believes it is
    in a venv it owns. NOTHING is written outside ``tmp_path``: the operator's
    real venvs and installs must never be touched by a test run.
    """
    prefix = _fake_venv(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    real = Path(os.path.realpath(sys.executable))
    if ".framework" in str(real):
        pytest.skip("framework interpreters cannot be branded (stub re-exec)")
    if real.stat().st_dev != os.stat(prefix).st_dev:
        pytest.skip("interpreter and tmp_path on different devices: hardlink impossible")
    link = procname.ensure_branded_interpreter()
    if link is None:
        pytest.skip("no branded image could be planted in this environment")
    return link


class TestStaleness:
    """The three triggers, each demonstrated firing AND repairing."""

    def test_replants_when_link_names_a_different_inode(self, branded, tmp_path):
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
        branded.unlink()
        os.link(stale, branded)  # different inode, but nlink == 2
        assert os.stat(branded).st_ino != real_ino
        assert os.stat(branded).st_nlink >= 2, "only trigger (a) may detect this"

        assert procname.ensure_branded_interpreter() == branded
        assert os.stat(branded).st_ino == real_ino

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
        assert os.stat(branded).st_nlink >= 2

    def test_replants_when_libpython_symlink_dangles(self, branded):
        """Trigger (c): the measured 100%-abort mode.

        Without ``<venv>/lib/libpython3.X.dylib`` the branded image dies with
        ``Abort trap: 6`` on every launch — 40/40 measured. A first run can
        spuriously succeed from a warm dyld launch closure, which is exactly why
        this is pinned by a test rather than by a manual smoke check.
        """
        name = procname._libpython_name()
        assert name, "a brandable interpreter must expose an LDLIBRARY dylib"
        lib = branded.parent.parent / "lib" / name
        assert lib.is_symlink() and lib.exists(), "plant must create a live symlink"

        lib.unlink()
        lib.symlink_to("/nonexistent/libpython.dylib")
        assert not lib.exists(), "precondition: the symlink dangles"

        assert procname.ensure_branded_interpreter() == branded
        assert lib.exists(), "the dangling symlink must be repaired"

    def test_orphan_temps_from_dead_plants_are_swept(self, branded):
        """A plant killed between ``os.link`` and ``os.replace`` leaks a temp.

        Reproduced with ``kill -9``: the cleanup on the error path cannot run,
        so one entry accumulates per crashed startup. A LIVE pid's temp must
        survive — a concurrent worktree is plausibly mid-plant.
        """
        bin_dir = branded.parent
        dead = bin_dir / f".{procname.BRAND}.999999.tmp"
        # A DIFFERENT live pid: this process's own temp name is the one the
        # plant itself claims and consumes via `os.replace`, so using it here
        # would test the plant's own bookkeeping rather than the sweep.
        mine = bin_dir / f".{procname.BRAND}.{os.getppid()}.tmp"
        dead.write_bytes(b"orphan")
        mine.write_bytes(b"in flight")

        branded.unlink()  # force the replant path, which is where the sweep runs
        assert procname.ensure_branded_interpreter() == branded

        assert not dead.exists(), "a dead pid's temp must be swept"
        assert mine.exists(), "a live pid's temp must be left alone"

    def test_replants_when_libpython_points_at_the_wrong_library(self, branded):
        """Trigger (c), second half: a LIVE symlink is not automatically CORRECT.

        Found while verifying the B4 fix. `exists()` only proves the link
        resolves, so after an interpreter upgrade a link aimed at the PREVIOUS
        install's dylib passed the check — and loading a mismatched libpython
        beside a fresh interpreter is the crash this shape exists to avoid.
        """
        name = procname._libpython_name()
        assert name
        lib = branded.parent.parent / "lib" / name
        lib.unlink()
        lib.symlink_to("/etc/hosts")  # live, resolvable, and wrong
        assert lib.exists(), "precondition: the symlink resolves"

        assert procname.ensure_branded_interpreter() == branded
        expected = Path(os.path.realpath(sys.executable)).parent.parent / "lib" / name
        assert lib.resolve() == expected.resolve()

    def test_orphan_temps_are_swept_in_lib_as_well_as_bin(self, branded):
        """REGRESSION (review round 1, B3): both plant directories are swept.

        A killed plant leaks a temp named for the BRAND in `bin/` or for the
        DYLIB in `lib/`. Sweeping only `bin/` left lib temps accumulating
        forever, one per crashed startup.
        """
        name = procname._libpython_name()
        assert name
        bin_orphan = branded.parent / f".{procname.BRAND}.999999.tmp"
        lib_orphan = branded.parent.parent / "lib" / f".{name}.999998.tmp"
        bin_orphan.write_bytes(b"orphan")
        lib_orphan.write_bytes(b"orphan")

        branded.unlink()  # force the replant path
        assert procname.ensure_branded_interpreter() == branded

        assert not bin_orphan.exists(), "bin/ orphan must be swept"
        assert not lib_orphan.exists(), "lib/ orphan must be swept too"

    def test_unreadable_libpython_parent_does_not_raise(self, branded, tmp_path):
        """QA round 2, Q3: `_needs_replant` is contractually no-raise.

        `Path.exists()` is not total — an unreadable parent directory makes it
        raise `PermissionError` (reproduced), which escaped a function this
        module documents as never raising. "Replant" is the safe answer: a
        libpython we cannot stat is not one to assume correct.
        """
        walled = tmp_path / "lib" / "sub"
        walled.mkdir(parents=True)
        dylib = walled / "libpython3.12.dylib"
        dylib.touch()
        walled.chmod(0o000)
        try:
            assert (
                procname._needs_replant(branded, Path(os.path.realpath(sys.executable)), dylib)
                is True
            )
        finally:
            # Restore so tmp_path teardown can remove the tree.
            walled.chmod(0o755)

    def test_healthy_shape_is_not_rewritten(self, branded):
        """The common path does no filesystem writes.

        This runs on every startup, so a plant that re-links unconditionally
        would churn an inode (and race sibling worktrees) for nothing.
        """
        before = os.stat(branded)
        assert procname.ensure_branded_interpreter() == branded
        after = os.stat(branded)
        assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


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

    def test_no_dylib_means_no_plant(self, tmp_path, monkeypatch):
        """Without a dylib to symlink, a planted hardlink would be a landmine.

        Refusing is correct: the alternative plants an image that aborts on
        every launch once dyld's warm closure expires.
        """
        monkeypatch.setattr(sys, "prefix", str(_fake_venv(tmp_path)))
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
        monkeypatch.setattr(procname, "_libpython_name", lambda: None)
        assert procname.ensure_branded_interpreter() is None
        assert not (tmp_path / "bin" / procname.BRAND).exists()

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
        test failing anywhere else.
        """
        from local_operator.browser_bridge import install as browser_install
        from local_operator.mobile import install as mobile_install
        from local_operator.wakes import install as wakes_install

        sentinel = tmp_path / "bin" / procname.BRAND
        sentinel.parent.mkdir(parents=True)
        sentinel.touch()
        monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: sentinel)

        # `render_plist` is typed `dict[str, object]`, so the element type is
        # narrowed here rather than indexed straight off an `object`.
        for rendered in (
            mobile_install.render_plist(1),
            browser_install.render_plist(1),
            wakes_install.render_plist(tmp_path),
        ):
            program = rendered["ProgramArguments"]
            assert isinstance(program, list)
            assert program[0] == str(sentinel)


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

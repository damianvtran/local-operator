"""The generation layout: one tree per build, and a pointer resolved at exec.

This is Phase 1 of the fix for 2026-09-15, when ``uv tool install --force``
recreated ``~/.local/share/uv/tools/local-operator`` in place while ~24 live
runtimes imported from it: 36 sessions died with no exit record and 113 crash
reports named the planted libpython dylib. The layout, its retention policy and
the atomic flip are all here in ``local_operator.update``; what these tests pin
is the set of properties the incident turned on, each of them a property a
future edit could quietly remove:

* the flip is atomic and never leaves ``current`` dangling;
* an install lands in its OWN generation (the per-generation uv environment is
  the mechanism, so the argv alone is not enough to assert);
* pruning cannot delete the tree the pointer names or one a live record names;
* the disk read follows the POINTER while the boot read stays this process's
  own tree — the distinction the whole design rests on;
* a spawn resolves the pointer's interpreter concretely, and falls back to
  ``sys.executable`` when there is no pointer to resolve.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from local_operator import update as update_mod

# ``_real`` is the non-raising realpath the layout itself compares with; these
# tests must compare the two spellings of one path the same way it does.
from local_operator.update import BuildStamp, UpdateError, _real

NEW = BuildStamp(version="0.52.0", source_ref="bbbbbbb2222")
OLD = BuildStamp(version="0.51.9", source_ref="aaaaaaa1111")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated ``$HOME``, which is what relocates the whole layout.

    ``stable_root()`` derives from ``Path.home()`` at every call precisely so
    this works: a module-level absolute path would write into the operator's
    real home from inside a test, and these tests build and delete installs.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LOP_INSTALL_ROOT", raising=False)
    return tmp_path


class FakeUv:
    """Stand in for ``uv tool install``, and record what it was asked to do.

    It builds the layout uv really builds — the venv, the console-script shims,
    the distribution metadata with its console-script entry points, and the
    interpreter — because several properties under test are about what the
    NEXT reader finds there (the launchers, the shim's fallback, the version
    ``disk_build`` reports) rather than about uv itself.
    """

    def __init__(self, version: str = "0.52.0", exit_code: int = 0) -> None:
        self.version = version
        self.exit_code = exit_code
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, argv: list[str], env: dict[str, str]) -> int:
        self.calls.append((argv, env))
        if self.exit_code == 0:
            _build_tree(
                Path(env["UV_TOOL_DIR"]) / "local-operator",
                Path(env["UV_TOOL_BIN_DIR"]),
                self.version,
            )
        return self.exit_code


def _build_tree(venv: Path, bin_dir: Path, version: str) -> None:
    """The files uv would leave for one generation, including its scripts.

    The console scripts are the shape ``uv tool install`` really writes — an
    ABSOLUTE shebang naming the venv they were installed into, plus a Python body
    — and they print the ``sys.prefix`` they ended up on. That is what makes them
    exec-able evidence: running one through the launcher chain reports which
    INTERPRETER actually answered, which is the only way the migration's
    shebang rebinding (:func:`local_operator.update._rebind_scripts`) can be
    asserted at all (review round 1, R-1/R-5: a fixture whose scripts said
    ``#!/bin/sh`` could not represent a migrated tree, so the test that should
    have caught the blocker passed for a tree that could not exhibit it).
    """
    for directory in (venv / "bin", venv / "lib" / "python3.12" / "site-packages", bin_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text("home = /nonexistent\n", encoding="utf-8")
    interpreter = venv / "bin" / "python3"
    if not interpreter.exists():
        # A symlink to the test process's own interpreter, so the scripts below
        # can actually run: ``pyvenv.cfg`` beside it is what makes the child
        # report THIS venv as its prefix.
        os.symlink(sys.executable, interpreter)
    for name in ("lop", "local-operator"):
        script = venv / "bin" / name
        script.write_text(
            f"#!{interpreter}\n" "import sys\n" "print('PREFIX', sys.prefix)\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        os.symlink(script, bin_dir / name)
    dist = venv / "lib" / "python3.12" / "site-packages" / f"local_operator-{version}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: local-operator\nVersion: {version}\n", encoding="utf-8"
    )
    entry_points = "[console_scripts]\n" + "".join(
        f"{name} = local_operator.cli:main\n" for name in ("lop", "local-operator")
    )
    (dist / "entry_points.txt").write_text(entry_points, encoding="utf-8")


#: What the child prints: the same call a runtime makes to stamp
#: ``SessionRecord.install_root``. Run in a child rather than imported here
#: because the value under test is what a RUNNING process reports.
_REPORT_INSTALL_ROOT = (
    "from local_operator.update import process_install_root as reported; print(reported())"
)


def _real_generation(name: str, version: str = "0.55.10") -> Path:
    """A generation whose venv directory is REAL, as a real install's is.

    ``_build_tree`` writes the tree uv would leave — which includes a real
    ``pyvenv.cfg`` and ``bin/python3`` — and its ``site-packages`` is then
    replaced by a symlink to this checkout's, so a child launched from here
    imports this distribution and its dependencies the way a real one does. The
    VENV DIRECTORY itself is what must stay real: ``process_install_root()``
    resolves symlinks, and a generation reached through one reports the tree it
    points at instead of itself (review round 2, R2-1).
    """
    generation = update_mod.generations_dir() / name
    install_root = generation / "tools" / "local-operator"
    _build_tree(install_root, generation / "bin", version)
    site_packages = next(install_root.glob("lib/python*/site-packages"))
    shutil.rmtree(site_packages)
    os.symlink(next(Path(sys.prefix).glob("lib/python*/site-packages")), site_packages)
    return install_root


def _reported_root(interpreter: Path, cwd: Path) -> Path:
    """``process_install_root()`` exactly as a child of ``interpreter`` reports it.

    ``__PYVENV_LAUNCHER__`` is stripped unconditionally rather than checked for:
    this machine's ``lop`` exports it to its children, and an inherited value
    makes a macOS child report the launcher's prefix instead of its own
    (``docs/design-install-generations.md`` §3.2). ``cwd`` is the temp directory
    the caller owns, so neither child can satisfy the import by accident through
    ``sys.path[0]`` pointing at the checkout.
    """
    env = {key: value for key, value in os.environ.items() if key != "__PYVENV_LAUNCHER__"}
    done = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [str(interpreter), "-c", _REPORT_INSTALL_ROOT],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
        cwd=cwd,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return Path(done.stdout.strip())


class TestTheRecordedInstallRoot:
    """``install_root`` is evidence only when the generation it names is real.

    The e2e busy-runtime cell asserts its premise from ``SessionRecord.install_root``
    — the child's own ``sys.prefix`` — because that is the one reading a
    mis-launched child cannot fake. It can only fail for that, though, if the two
    launch paths report DIFFERENT roots, which is the property this test measures
    directly and cheaply: two short-lived interpreters, no TUI, no daemon.
    """

    def test_a_real_generation_reports_itself_where_the_checkout_does_not(
        self, home: Path, tmp_path: Path
    ) -> None:
        install_root = _real_generation("20260101T000000Z-old")
        from_generation = _reported_root(install_root / "bin" / "python3", tmp_path)
        from_checkout = _reported_root(Path(sys.executable), tmp_path)
        assert from_generation == install_root.resolve(), from_generation
        assert from_checkout == Path(sys.prefix).resolve(), from_checkout
        assert from_generation != from_checkout, (
            "a generation whose venv is a symlink into the checkout resolves to the "
            "same tree whichever interpreter started the child, so the e2e premise "
            "assertion would pass for a mis-launched child (R2-1)"
        )


def _install(version: str = "0.52.0", commit: str = "", ref: str = "") -> Path:
    """One generation, built by the fake uv, with the pointer flipped onto it."""
    return update_mod.install_into_generation(
        runner=FakeUv(version=version),
        version=version,
        commit=commit,
        ref=ref,
        origin=update_mod.SNAPSHOT_SOURCE_TOKEN if commit else update_mod.PYPI_SOURCE_TOKEN,
    )


# -- the flip ------------------------------------------------------------------


class TestTheFlip:
    def test_a_flip_creates_the_pointer_and_names_the_generation(self, home: Path) -> None:
        generation = _install("0.52.0")
        assert update_mod.pointer_path().is_symlink()
        assert update_mod.current_generation() == generation.resolve()
        assert (
            update_mod.current_install_root() == (generation / "tools" / "local-operator").resolve()
        )

    def test_the_spawn_path_never_names_the_mutable_pointer(self, home: Path) -> None:
        """A child handed ``<pointer>/bin/python3`` imports through ``current``.

        That is the original failure wearing a new hat: CPython resolves a venv
        from the directory of the path it was invoked by, so launching the
        pointer gives a process whose ``sys.path`` a later flip redirects. The
        interpreter this layout offers is built from the RESOLVED pointer, and
        this pins it.
        """
        _install("0.52.0")
        interpreter = update_mod.current_interpreter()
        assert interpreter is not None
        assert "current" not in str(interpreter)
        assert interpreter.parent.parent == update_mod.current_install_root()

    def test_a_flip_replaces_an_existing_pointer(self, home: Path) -> None:
        first = _install("0.51.9")
        second = _install("0.52.0")
        assert first != second
        assert update_mod.current_generation() == second.resolve()

    def test_a_flip_refuses_a_generation_that_is_not_there(self, home: Path) -> None:
        """The one state every reader of this layout must never observe."""
        update_mod.generations_dir().mkdir(parents=True)
        with pytest.raises(UpdateError, match="missing generation"):
            update_mod.flip_pointer(update_mod.generations_dir() / "nope")

    def test_the_pointer_still_names_the_old_generation_until_the_rename(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE ATOMICITY CLAIM, deterministically: a rename, never an unlink.

        Observed at the last instant before the swap — the moment a concurrent
        reader would be looking. A ``current`` written as unlink-then-create is
        absent at exactly this point, which is the failure a ``lop`` exec would
        report as "No such file or directory"; a staging symlink plus
        ``os.rename`` leaves the old generation named until the replacement
        lands.
        """
        first = _install("0.51.9")
        second = _install("0.52.0")
        observed: list[Path | None] = []
        real_rename = os.rename

        def _observe(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
            if str(dst) == str(update_mod.pointer_path()):
                observed.append(update_mod.current_generation())
            real_rename(src, dst)

        monkeypatch.setattr(update_mod.os, "rename", _observe)
        update_mod.flip_pointer(first)
        assert observed == [second]
        assert update_mod.current_generation() == first

    def test_a_racing_reader_never_lands_on_a_generation_that_is_not_there(
        self, home: Path
    ) -> None:
        """Under continuous flipping, every read is a REAL generation or a
        TRANSIENT hiccup — never a pointer that names nothing.

        The distinction is the guarantee. ``os.rename`` makes the pointer always
        name a live generation, so a read that fails (macOS raises EINVAL on
        both ``readlink`` and ``realpath`` when the link is replaced underneath
        it — measured by this test) must be followed by a successful one; a
        pointer left dangling by an unlink-then-create would fail the retry
        immediately, and it is that retry, not the first read, that decides.
        """
        generations = [_install(f"0.52.{index}") for index in range(4)]
        known = {_real(path) for path in generations}
        pointer = update_mod.pointer_path()
        staged_path = pointer.with_name("current.staged")
        failures: list[str] = []
        flips = 0
        stop = threading.Event()

        def _flip() -> None:
            """The SAME shape ``flip_pointer`` uses: stage a sibling, rename."""
            nonlocal flips
            index = 0
            while not stop.is_set():
                try:
                    staged_path.unlink(missing_ok=True)
                    os.symlink(generations[index % len(generations)], staged_path)
                    os.rename(staged_path, pointer)
                except OSError as exc:  # pragma: no cover — only on a broken flip
                    failures.append(f"flip: {exc}")
                    return
                index += 1
                flips += 1

        reader = threading.Thread(target=_flip, daemon=True)
        reader.start()
        seen = 0
        hiccups = 0
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    target = update_mod.current_generation()
                except OSError:
                    hiccups += 1
                    continue
                if target is None:
                    hiccups += 1
                    if update_mod.current_generation() is None:
                        failures.append("read: the pointer names nothing, twice")
                        break
                    continue
                if _real(target) not in known:  # pragma: no cover — a third generation
                    failures.append(f"read: {target} is not one of the four")
                    break
                seen += 1
        finally:
            stop.set()
            reader.join(timeout=5)
            update_mod.flip_pointer(generations[0])
        # CONCURRENCY, not throughput: the counts only have to show that both
        # sides really ran against each other (the reader is GIL-bound by the
        # flipping thread, so a per-second rate here would be measuring the
        # host rather than the flip).
        assert flips > 5, f"the flipper never got going ({flips})"
        assert seen > 5, f"the reader never got a turn ({seen})"
        assert failures == []

    def test_the_flip_stages_beside_the_pointer(self, home: Path) -> None:
        """A staging name in another directory could cross a filesystem.

        ``os.rename`` is only atomic within one, so the temp symlink has to be a
        SIBLING of ``current`` — a property worth pinning because a "tidier"
        temp dir is an easy edit with no visible symptom until a host has the
        layout on two mounts.
        """
        generation = _install("0.52.0")
        staged: list[Path] = []
        real_symlink = os.symlink

        def _record(target: str | os.PathLike[str], link: str | os.PathLike[str]) -> None:
            staged.append(Path(str(link)))
            real_symlink(target, link)

        original = update_mod.os.symlink
        update_mod.os.symlink = _record  # type: ignore[assignment]
        try:
            update_mod.flip_pointer(generation)
        finally:
            update_mod.os.symlink = original  # type: ignore[assignment]
        assert staged, "no staged symlink was written"
        assert all(path.parent == update_mod.pointer_path().parent for path in staged)


# -- the install ---------------------------------------------------------------


class TestInstallIntoGeneration:
    def test_the_environment_aims_uv_at_one_generation(self, home: Path) -> None:
        """``UV_TOOL_DIR``/``UV_TOOL_BIN_DIR`` are the mechanism, not decoration.

        Without them uv installs into the shared default tool dir and rewrites
        the tree the running fleet imports from — the incident this layout
        exists to end. They are per-generation and absolute.
        """
        fake = FakeUv(version="0.52.0")
        generation = update_mod.install_into_generation(runner=fake, version="0.52.0")
        assert len(fake.calls) == 1
        argv, env = fake.calls[0]
        assert argv == ["uv", "tool", "install", "--force", "local-operator"]
        assert env["UV_TOOL_DIR"] == str(generation / "tools")
        assert env["UV_TOOL_BIN_DIR"] == str(generation / "bin")
        # The real ~/.local/bin must never be uv's target: its entries are this
        # layout's stable launchers.
        assert env["UV_TOOL_BIN_DIR"] != str(Path.home() / ".local" / "bin")

    def test_a_snapshot_source_rides_as_from(self, home: Path) -> None:
        fake = FakeUv(version="0.52.0")
        update_mod.install_into_generation(
            "/tmp/some-tree", runner=fake, version="0.52.0", commit="a" * 40, ref="main"
        )
        argv, _env = fake.calls[0]
        assert argv == [
            "uv",
            "tool",
            "install",
            "--force",
            "--from",
            "/tmp/some-tree",
            "local-operator",
        ]

    def test_the_marker_is_written_before_the_pointer_moves(self, home: Path) -> None:
        language = _install("0.52.0", commit="c" * 40, ref="main")
        marker = language / "tools" / "local-operator" / ".lop-source"
        assert marker.read_text(encoding="utf-8") == f"{'c' * 40} main\n"
        assert update_mod.current_generation() == language.resolve()

    def test_a_failed_install_leaves_nothing_behind_and_no_flip(self, home: Path) -> None:
        good = _install("0.51.9")
        with pytest.raises(UpdateError, match="exited 9"):
            update_mod.install_into_generation(runner=FakeUv(exit_code=9), version="0.52.0")
        assert update_mod.current_generation() == good.resolve()
        assert sorted(path.name for path in update_mod.generations_dir().iterdir()) == [good.name]

    def test_a_raising_runner_is_reported_as_an_install_failure(self, home: Path) -> None:
        def _explode(_argv: list[str], _env: dict[str, str]) -> int:
            raise OSError("uv is not on PATH")

        with pytest.raises(UpdateError):
            update_mod.install_into_generation(runner=_explode, version="0.52.0")
        assert list(update_mod.generations_dir().iterdir()) == []

    def test_a_script_that_cannot_be_repointed_fails_the_migration(
        self, home: Path, tmp_path: Path
    ) -> None:
        """R2-4: the migration's one promise is that the copy runs from the copy.

        A copy whose ``lop`` could not be re-pointed must not flip the pointer and
        report success: the operator would be left running the legacy venv through
        a "successful" migration, which is R-1's symptom with a success message.
        The copy is removed on the way out, so the machine is as it was.

        Permissions, rather than a monkeypatched ``_rebind_scripts``, because the
        failure being tested is exactly the one a real migration meets.
        """
        if getattr(os, "geteuid", lambda: 1)() == 0:  # pragma: no cover — root ignores the mode
            pytest.skip("file modes do not bind this process")
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        os.chmod(legacy / "bin" / "lop", 0o000)
        with pytest.raises(UpdateError) as refused:
            update_mod.clone_into_generation(legacy)
        assert "lop" in str(refused.value), refused.value
        assert not update_mod.pointer_path().is_symlink(), "the pointer must not move"
        assert not list(update_mod.generations_dir().glob("*")), "no half-built generation"

    def test_a_flip_that_cannot_happen_is_a_refusal_and_leaves_no_tree(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R-9: an unwritable stable root must not raise a bare ``OSError``.

        The flip used to sit OUTSIDE the handler that removes a tree on failure,
        so ``EACCES``/``ENOSPC`` escaped as ``OSError`` — not the "one refusal
        sentence" the neighbouring comment promises — and left a fully built,
        marker-bearing generation that no reader references. The pointer itself
        is untouched either way (``os.rename`` either happened or it did not),
        which is the property asserted here alongside the tree being gone.
        """
        good = _install("0.51.9")

        def _boom(_generation: Path) -> None:
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(update_mod, "flip_pointer", _boom)
        with pytest.raises(UpdateError, match="could not point current"):
            _install("0.52.0")
        assert update_mod.current_generation() == good.resolve()
        assert sorted(path.name for path in update_mod.generations_dir().iterdir()) == [good.name]

    def test_two_installs_get_two_generations(self, home: Path) -> None:
        first = _install("0.52.0")
        second = _install("0.52.0")
        assert first != second
        assert update_mod.current_generation() == second.resolve()

    def test_the_stable_launchers_name_the_pointer(self, home: Path) -> None:
        _install("0.52.0")
        launcher = Path.home() / ".local" / "bin" / "lop"
        assert launcher.is_symlink()
        assert os.readlink(launcher) == str(update_mod.pointer_path() / "bin" / "lop")
        # It must EXECUTE: the chain resolves to the generation's own shim.
        generation = update_mod.current_generation()
        assert generation is not None
        assert launcher.resolve() == (generation / "bin" / "lop").resolve()

    def test_the_daemon_shim_resolves_the_pointer_at_exec(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shim is what a launchd/systemd unit names, and it is a SCRIPT.

        A symlink at that path loses the venv entirely (CPython decides "am I in
        a venv" from the parent directory of the path it was executed through —
        verified, it resolves to the base interpreter with no site-packages), so
        the shape is load-bearing rather than stylistic.

        ``process_install_root`` is monkeypatched because the whole daemon-image
        offer is gated on THIS process running from a generation: the test
        process runs from a worktree venv, which is exactly the case that must
        keep its pre-generation plists.
        """
        generation = _install("0.52.0")
        monkeypatch.setattr(
            update_mod,
            "process_install_root",
            lambda: str(generation / "tools" / "local-operator"),
        )
        shim = update_mod.ensure_daemon_image()
        assert shim == update_mod.daemon_image_path()
        assert shim is not None and shim.is_file() and os.access(shim, os.X_OK)
        body = shim.read_text(encoding="utf-8")
        assert "pwd -P" in body, "the pointer must be resolved, not followed in place"
        assert update_mod.daemon_image() == shim
        # Point the pointer elsewhere and the SAME file resolves the new tree.
        second = _install("0.52.1")
        assert second != generation
        probe = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(shim), "-c", "import sys; print(sys.prefix)"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert probe.returncode == 0, probe.stderr
        assert (
            Path(probe.stdout.strip()).resolve() == (second / "tools" / "local-operator").resolve()
        )

    def test_no_daemon_shim_without_a_generation_layout(self, home: Path) -> None:
        """A checkout or a pip install must keep its pre-generation plists."""
        assert update_mod.ensure_daemon_image() is None
        assert update_mod.daemon_image() is None


# -- the disk read -------------------------------------------------------------


class TestDiskBuild:
    def test_the_pointer_answers_not_this_process(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install("0.52.0")
        monkeypatch.setattr(update_mod, "install_kind", lambda **_k: update_mod.InstallKind.UV_TOOL)
        assert update_mod.disk_build() == BuildStamp(version="0.52.0", source_ref="")

    def test_a_second_generation_moves_the_disk_build(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install("0.51.9")
        monkeypatch.setattr(update_mod, "install_kind", lambda **_k: update_mod.InstallKind.UV_TOOL)
        assert update_mod.disk_build() == BuildStamp(version="0.51.9", source_ref="")
        _install("0.52.0")
        assert update_mod.disk_build() == BuildStamp(version="0.52.0", source_ref="")

    def test_a_source_checkout_has_no_disk_build(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """design-build-skew §6.5: a worktree must not retire on the global pointer."""
        _install("0.52.0")
        monkeypatch.setattr(
            update_mod, "install_kind", lambda **_k: update_mod.InstallKind.EDITABLE
        )
        assert update_mod.disk_build() is None

    def test_an_explicit_root_is_read_wholesale(self, home: Path) -> None:
        """The e2e seam: a tree with metadata answers with ITS version and ref."""
        generation = _install("0.52.0", commit="d" * 40, ref="main")
        assert update_mod.disk_build(generation / "tools" / "local-operator") == BuildStamp(
            version="0.52.0", source_ref="d" * 40
        )

    def test_an_explicit_root_without_metadata_keeps_the_running_version(
        self, home: Path, tmp_path: Path
    ) -> None:
        """``LOP_BUILD_PREFIX``'s shape: a temp dir carrying only a marker."""
        fake = tmp_path / "fake"
        fake.mkdir()
        (fake / ".lop-source").write_text("e" * 40 + " main\n", encoding="utf-8")
        stamp = update_mod.disk_build(fake)
        assert stamp is not None
        assert stamp.source_ref == "e" * 40
        assert stamp.version == update_mod.installed_version()

    def test_an_unreadable_pointer_is_no_answer(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(update_mod, "install_kind", lambda **_k: update_mod.InstallKind.UV_TOOL)
        assert update_mod.disk_build() is None

    def test_installed_build_still_describes_this_process(self, home: Path) -> None:
        """``lop --version`` must not start reporting the pointer's build."""
        _install("0.52.0")
        assert update_mod.installed_build().version == update_mod.installed_version()


# -- pruning -------------------------------------------------------------------


class TestPruning:
    def test_the_pointer_target_is_never_removed(self, home: Path) -> None:
        generations = [_install(f"0.52.{index}") for index in range(5)]
        removed = update_mod.prune_generations()
        assert update_mod.current_generation() == generations[-1].resolve()
        assert generations[-1] not in removed

    def test_a_live_record_holds_its_generation(self, home: Path) -> None:
        """The one thing that makes pruning safe on a busy machine."""
        generations = [_install(f"0.52.{index}") for index in range(5)]
        held = generations[0]
        removed = update_mod.prune_generations(
            referenced=[held / "tools" / "local-operator"],
        )
        assert held not in removed
        assert held.is_dir()
        assert not any(path == held for path in removed)

    def test_the_last_two_unreferenced_generations_survive(self, home: Path) -> None:
        generations = [_install(f"0.52.{index}") for index in range(5)]
        removed = update_mod.prune_generations()
        survivors = [path for path in generations if path.is_dir()]
        assert len(survivors) == update_mod.DEFAULT_KEEP_GENERATIONS + 1  # + the pointer's
        assert generations[0] in removed
        assert generations[1] in removed
        assert generations[-1].is_dir()

    def test_an_install_in_flight_is_not_pruned(self, home: Path) -> None:
        """No marker means uv is still writing there; only age calls it debris."""
        _install("0.52.0")
        in_flight = update_mod.generations_dir() / "20260101T000000Z-writing"
        in_flight.mkdir()
        (in_flight / "tools" / "local-operator").mkdir(parents=True)
        assert update_mod.prune_generations(keep=0).count(in_flight) == 0
        assert in_flight.is_dir()

    def test_an_abandoned_tree_is_reclaimed_once_it_is_old(self, home: Path) -> None:
        _install("0.52.0")
        debris = update_mod.generations_dir() / "20260101T000000Z-crashed"
        debris.mkdir()
        (debris / "tools" / "local-operator").mkdir(parents=True)
        removed = update_mod.prune_generations(
            keep=0, now=time.time() + update_mod._PARTIAL_TTL_S + 1
        )
        assert debris in removed
        assert not debris.exists()

    def test_an_interrupted_flips_staging_link_is_swept(self, home: Path) -> None:
        """R-6: a ``kill -9`` between ``os.symlink`` and ``os.rename``.

        ``flip_pointer`` unlinks its own staging name in a ``finally``, which
        covers failures inside the call but not a death mid-call, and nothing
        else reclaims one — the stable root is meant to hold a closed set
        (``current``, ``bin/``, ``generations/``). Aged like every other in-flight
        artefact, so a CONCURRENT flip between its two steps is never deleted
        underneath.
        """
        generation = _install("0.52.0")
        pointer = update_mod.pointer_path()
        stale = pointer.with_name(f"{pointer.name}.tmp-999999")
        fresh = pointer.with_name(f"{pointer.name}.tmp-999998")
        for link in (stale, fresh):
            os.symlink(generation, link)
        old = time.time() - update_mod._PARTIAL_TTL_S - 60
        os.utime(stale, (old, old), follow_symlinks=False)

        update_mod.prune_generations(now=time.time())
        assert not stale.exists(), "an interrupted flip left litter in the stable root"
        assert fresh.is_symlink(), "a concurrent flip's staging link must survive"

    def test_pruning_a_machine_with_no_layout_is_a_no_op(self, home: Path) -> None:
        assert update_mod.prune_generations() == []


# -- migration -----------------------------------------------------------------


class TestMigration:
    def test_the_migrated_launcher_runs_the_generation_not_the_source(
        self, home: Path, tmp_path: Path
    ) -> None:
        """R-1: a verbatim copy keeps the SOURCE tree's absolute shebang.

        Executed, not inspected — the console scripts in the fixture print the
        ``sys.prefix`` they ended up on, so this asserts WHICH INTERPRETER
        answered. Before the fix the launcher chain
        (``~/.local/bin/lop`` → ``current/bin/lop`` → ``<gen>/bin/lop``) ended at
        a script whose ``#!`` named the legacy venv, and the prefix printed here
        was the SOURCE tree — the one the host rewrites in place, which is the
        whole hazard. This test fails against that tree, which the previous
        version could not: it asserted a resolved path was a file and never ran
        anything, so a launcher pointing at a foreign interpreter passed it
        (review round 1, R-1/R-5).
        """
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        generation = update_mod.clone_into_generation(legacy)
        launcher = Path.home() / ".local" / "bin" / "lop"

        ran = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(launcher)], capture_output=True, text=True, timeout=60
        )
        assert ran.returncode == 0, ran.stdout + ran.stderr
        reported = Path(ran.stdout.split()[-1])
        assert reported.resolve() == (generation / "tools" / "local-operator").resolve()
        assert reported.resolve() != legacy.resolve(), "the copy still ran the source tree"

    def test_the_migration_plants_the_daemon_shim(
        self, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R-2: the migrating process IS the legacy tree, so the this-process
        gate answered ``None`` and no shim was ever written — QA measured
        ``<stable>/bin/python3`` missing right after a real ``lop install
        migrate``, and every plist rendered afterwards kept naming a path inside
        the legacy venv.

        ``install_kind`` is monkeypatched because the process running this test
        is a source checkout, which deliberately never names a machine-level
        artefact (``_may_name_the_shim``, the ``_repair_refusal`` rule). A real
        migration runs from a uv-tool or pipx tree, and that is the read half
        this asserts works afterwards.
        """
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        update_mod.clone_into_generation(legacy)
        shim = update_mod.daemon_image_path()
        assert shim.is_file(), "the migration must leave the shim a plist would name"
        assert os.access(shim, os.X_OK)
        monkeypatch.setattr(update_mod, "install_kind", lambda **_k: update_mod.InstallKind.UV_TOOL)
        assert update_mod.daemon_image() == shim

    def test_the_legacy_tree_is_copied_not_moved(self, home: Path, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        (legacy / ".lop-source").write_text("f" * 40 + " main\n", encoding="utf-8")
        generation = update_mod.clone_into_generation(legacy)
        assert legacy.is_dir(), "the migration must not consume the install it came from"
        copied = generation / "tools" / "local-operator"
        assert (copied / ".lop-source").read_text(encoding="utf-8") == f"{'f' * 40} main\n"
        assert (copied / "pyvenv.cfg").is_file()
        assert update_mod.current_generation() == generation.resolve()
        stamp = update_mod.disk_build(copied)
        assert stamp is not None and stamp.source_ref == "f" * 40

    def test_a_migrated_generation_can_be_launched_through_the_pointer(
        self, home: Path, tmp_path: Path
    ) -> None:
        """The bug the live walkthrough found: a clone left ``<gen>/bin`` missing.

        No installer runs during a migration, so nothing creates that directory —
        and the stable launchers point THROUGH it
        (``~/.local/bin/lop -> <stable>/current/bin/lop``). Without it, migrating
        replaced a working ``lop`` with a dangling symlink while printing nothing
        but success: the unit tests passed, the copy printed, and
        ``lop --version`` said "No such file or directory".

        THIS TEST IS ABOUT THE CHAIN'S SHAPE, not about what the chain executes: it
        asserts the launcher resolves through the pointer to the generation's own
        script and that the script is a real file. The executing half — that the
        script then runs from the copy — is
        :meth:`test_the_migrated_launcher_runs_the_generation_not_the_source`, which
        reads the prefix the script reports (review round 2, R2-6).
        """
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        generation = update_mod.clone_into_generation(legacy)
        assert (generation / "bin" / "lop").is_symlink()
        assert (generation / "bin" / "local-operator").is_symlink()
        launcher = Path.home() / ".local" / "bin" / "lop"
        assert launcher.resolve() == (generation / "bin" / "lop").resolve()
        assert launcher.resolve().is_file(), "the chain must end at a real script"

    def test_a_launcher_is_never_pointed_at_a_missing_script(
        self, home: Path, tmp_path: Path
    ) -> None:
        """A dangling launcher is worse than a stale one, so the link is skipped.

        This is the guard that turns the migration bug above from silent into
        visible: the launcher that already worked is left alone rather than
        replaced with a symlink to nothing.
        """
        generation = _install("0.52.0")
        launcher = Path.home() / ".local" / "bin" / "lop"
        before = os.readlink(launcher)
        (generation / "bin" / "lop").unlink()
        written = update_mod.write_stable_launchers(generation)
        assert launcher not in written
        assert os.readlink(launcher) == before

    def test_a_tree_without_a_marker_gets_one(self, home: Path, tmp_path: Path) -> None:
        """Every finished generation carries a marker: pruning reads it."""
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        generation = update_mod.clone_into_generation(legacy)
        marker = generation / "tools" / "local-operator" / ".lop-source"
        assert marker.is_file()
        assert (
            marker.read_text(encoding="utf-8").strip()
            == f"{update_mod.SNAPSHOT_SOURCE_TOKEN} 0.51.9"
        )

    def test_a_directory_outside_the_layout_is_not_a_generation(self, home: Path) -> None:
        assert update_mod._is_generation_install(Path("/tmp/elsewhere/.venv")) is False

    def test_migrate_refuses_a_source_checkout(
        self, home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The migration copies ``sys.prefix`` and flips the machine's pointer.

        Run from a worktree venv it would aim the whole machine at a copy of a
        developer's checkout — the accident ``_repair_refusal`` already guards
        for the supervised daemons. Nothing is copied and nothing is flipped.
        """
        monkeypatch.setattr(
            update_mod, "install_kind", lambda **_k: update_mod.InstallKind.EDITABLE
        )
        assert update_mod.install_migrate_command() == 1
        assert "refusing to migrate" in capsys.readouterr().out
        assert not update_mod.generations_dir().exists()
        assert not update_mod.pointer_path().exists()

    def test_the_idempotent_branch_reports_rather_than_copying(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        generation = _install("0.52.0")
        monkeypatch.setattr(
            update_mod, "process_install_root", lambda: str(generation / "tools" / "local-operator")
        )
        assert update_mod._is_generation_install() is True
        assert update_mod.install_migrate_command() == 0
        assert len(list(update_mod.generations_dir().iterdir())) == 1


# -- snapshots -----------------------------------------------------------------


class TestSnapshotSource:
    def test_a_directory_is_used_as_it_stands(self, tmp_path: Path) -> None:
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "pyproject.toml").write_text('[project]\nversion = "0.52.0"\n', encoding="utf-8")
        (tree / ".git").mkdir()
        snapshot = update_mod.resolve_snapshot(str(tree))
        assert snapshot.path == tree.resolve()
        assert snapshot.version == "0.52.0"
        assert snapshot.temporary is False
        assert snapshot.label == str(tree.resolve())

    def test_a_ref_is_archived_out_of_the_repository(self, tmp_path: Path) -> None:
        """The ref half of the marker is why: two builds of one version differ."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text('[project]\nversion = "0.52.0"\n', encoding="utf-8")
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@e",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@e",
        }

        def _git(*args: str) -> None:
            subprocess.run(
                ["git", "-C", str(repo), *args], check=True, env=env, capture_output=True
            )

        _git("init", "-q")
        _git("add", ".")
        _git("commit", "-qm", "one")
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        cwd = os.getcwd()
        os.chdir(repo)
        try:
            snapshot = update_mod.resolve_snapshot("HEAD")
        finally:
            os.chdir(cwd)
        assert snapshot.commit == head
        assert snapshot.ref == "HEAD"
        assert snapshot.temporary is True
        assert (snapshot.path / "pyproject.toml").is_file()
        assert snapshot.path != repo

    def test_a_value_that_is_neither_is_a_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(UpdateError, match="neither a directory nor a git ref"):
            update_mod.resolve_snapshot("no-such-ref")


# -- the record's install root -------------------------------------------------


def test_the_record_names_its_install_root() -> None:
    """Additive: no PROTOCOL_VERSION bump, and an older record still loads."""
    from local_operator.session.runtime.types import PROTOCOL_VERSION, SessionRecord

    record = SessionRecord(
        pid=1,
        kind="tui",
        session_id="s",
        conversation_name="n",
        cwd="/",
        model_label="m",
        control_port=0,
        control_key="k",
        install_root="/x/tools/local-operator",
    )
    assert record.install_root == "/x/tools/local-operator"
    assert record.to_json()["install_root"] == "/x/tools/local-operator"
    without = {key: value for key, value in record.to_json().items() if key != "install_root"}
    assert SessionRecord.from_json(without).install_root == ""
    assert PROTOCOL_VERSION == 5, "an additive field must not move the wire version"


def test_referenced_roots_reads_session_and_serve_records(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pruning and the processes that publish records must agree."""
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    record = SessionRecord(
        pid=os.getpid(),
        kind="tui",
        session_id="s",
        conversation_name="n",
        cwd="/",
        model_label="m",
        control_port=0,
        control_key="k",
        install_root=str(
            home / ".local" / "share" / "lop" / "generations" / "g1" / "tools" / "local-operator"
        ),
    )
    registry.publish(record)
    try:
        roots = update_mod.referenced_install_roots()
    finally:
        registry.unpublish(record.pid)
    assert any("generations/g1" in str(path) for path in roots)


def test_referenced_roots_read_the_default_config_root_too(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-7: the ambient config root is not the machine's only one.

    ``registry.scan()`` reads ``config_dir()``, so a prune run under an isolated
    root — a QA pass, a second profile — could not see the sessions the DEFAULT
    root publishes, and their generations fell back to the count margin. Both
    roots are read now, and this is the test that would fail if the ambient one
    were read alone.
    """
    from local_operator.paths import config_dir
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    ambient = tmp_path / "ambient-config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(ambient))
    assert config_dir() == ambient

    generation = _install("0.52.0")
    default_root = Path.home() / ".local-operator"
    record = SessionRecord(
        pid=os.getpid(),
        kind="tui",
        session_id="r7",
        conversation_name="n",
        cwd="/",
        model_label="m",
        control_port=0,
        control_key="k",
        install_root=str(generation / "tools" / "local-operator"),
    )
    registry.publish(record, root=default_root)
    try:
        roots = update_mod.referenced_install_roots()
    finally:
        registry.unpublish(record.pid, default_root)

    assert any(
        Path(root).resolve() == (generation / "tools" / "local-operator").resolve()
        for root in roots
    ), roots

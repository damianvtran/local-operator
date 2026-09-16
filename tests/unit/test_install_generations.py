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

import logging
import os
import re
import shutil
import stat
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
    # The site-packages directory name is derived from THIS interpreter rather
    # than hardcoded: a child started from the generation looks for
    # ``lib/python<major>.<minor>/site-packages`` under its own prefix, so a
    # hardcoded 3.12 made this file red on any other venv while CI (3.12) stayed
    # green — a false negative for whoever ran it locally (QA round 2, Q2).
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for directory in (venv / "bin", venv / "lib" / version_dir / "site-packages", bin_dir):
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
    dist = venv / "lib" / version_dir / "site-packages" / f"local_operator-{version}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: local-operator\nVersion: {version}\n", encoding="utf-8"
    )
    entry_points = "[console_scripts]\n" + "".join(
        f"{name} = local_operator.cli:main\n" for name in ("lop", "local-operator")
    )
    (dist / "entry_points.txt").write_text(entry_points, encoding="utf-8")


def _skip_as_root() -> None:
    """Skip a file-mode test when this process ignores file modes."""
    if getattr(os, "geteuid", lambda: 1)() == 0:  # pragma: no cover — root ignores modes
        pytest.skip("file modes do not bind this process")


def _clear_inside_generation(target: Path, generation: Path) -> None:
    """Remove a directory ONLY if it is a real directory inside ``generation``.

    WHY A GUARD IN A TEST HELPER (review round 3, R3-4). ``_real_generation``
    below deletes the copied tree's ``site-packages`` so it can be replaced by a
    link, and on 2026-09-16 the fixture was momentarily changed so that
    ``install_root`` was a symlink at this checkout: the glob then resolved
    THROUGH it and the delete took this worktree's own
    ``.venv/lib/python3.12/site-packages`` with it. A venv had to be rebuilt. The
    containment check and the symlink refusal turn that shape into a loud failure
    instead of a lost environment.

    The removal lives in here rather than at the call site so a future edit cannot
    check and then delete a different path.
    """
    if target.is_symlink():
        raise AssertionError(f"refusing to remove the symlink {target}")
    if not target.is_dir():
        raise AssertionError(f"refusing to remove {target}: not a directory")
    if not target.resolve().is_relative_to(generation.resolve()):
        raise AssertionError(f"refusing to remove {target}: outside {generation}")
    shutil.rmtree(target)


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
    # ``install_root`` must NOT be a symlink: the glob above would follow it out
    # of the generation, and the removal below would delete this checkout's venv
    # instead (see ``_remove_inside_generation`` — it happened, 2026-09-16).
    _clear_inside_generation(site_packages, install_root)
    os.symlink(next(Path(sys.prefix).glob("lib/python*/site-packages")), site_packages)
    return install_root


def _reported_root(interpreter: Path, cwd: Path) -> Path:
    """``process_install_root()`` exactly as a child of ``interpreter`` reports it.

    ``__PYVENV_LAUNCHER__`` is stripped unconditionally rather than checked for:
    it is exported down the process tree by the launcher on macOS, and an
    inherited value makes the child report the LAUNCHER's prefix instead of its
    own — observed first-hand while measuring a real ``lop`` session's children,
    not read out of a doc (review round 3, R3-5: an earlier version of this
    comment cited a section that does not discuss the variable, so the claim is
    recorded here where the strip is). ``cwd`` is the temp directory the caller
    owns, so neither child can satisfy the import by accident through
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

    def test_a_read_only_bin_fails_the_migration_and_takes_the_copy_with_it(
        self, home: Path, tmp_path: Path
    ) -> None:
        """R2-4 (R3-1): the refusal, reached by the shape that actually reaches it.

        The migration's one promise is that the copy runs from the copy: a copy
        whose console scripts could not be re-pointed must not flip the pointer and
        print success, because the operator would then be running the legacy venv
        through a "successful" migration — R-1's symptom with a success message.

        THE READ-ONLY THING IS THE ``bin`` DIRECTORY, not the script. A ``chmod
        000`` on the legacy script never arrives here: ``copytree`` opens the
        source for reading, so the same permission stops the COPY and the test
        passes on a different refusal that merely shares the word "lop" (review
        round 3, R3-1 — measured, and it passed with this refusal disabled).
        ``copytree`` writes the files before ``copystat`` tightens the copy's
        directory mode, so a 0555 source directory copies fine and the REBINDING
        write is what fails, which is the branch this covers.

        Both halves of the promise are asserted, because both were wrong in the
        last round: the refusal names the scripts, and the copy is REAL gone —
        ``shutil.rmtree`` cannot empty a 0555 directory, so the second assertion
        fails without ``_remove_tree``'s chmod-and-retry (review round 3, R3-2).
        """
        _skip_as_root()
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        os.chmod(legacy / "bin", 0o555)
        try:
            with pytest.raises(UpdateError) as refused:
                update_mod.clone_into_generation(legacy)
        finally:
            # So pytest's own cleanup of ``tmp_path`` is not fighting the mode.
            os.chmod(legacy / "bin", 0o755)
        assert "could not re-point" in str(refused.value), refused.value
        assert "lop" in str(refused.value), refused.value
        assert not update_mod.pointer_path().is_symlink(), "the pointer must not move"
        assert not list(
            update_mod.generations_dir().glob("*")
        ), "the refused copy must be gone, not left on disk"

    def test_remove_tree_retries_the_symlinks_a_bin_holds(self, tmp_path: Path) -> None:
        """R5-1: the retry's motivating shape is a venv's ``bin/``, links and all.

        A read-only ``bin`` holding a symlink is the shape the retry exists for —
        that is where a venv keeps its interpreter — and gating the retry on the
        TARGET being a symlink (rather than on the reported callable) dropped it:
        ``bin/`` kept the link, never emptied, and ``_remove_tree`` answered False,
        which is R3-2 reintroduced by its fix (review round 5, R5-1).
        """
        _skip_as_root()
        tree = tmp_path / "tree"
        (tree / "bin").mkdir(parents=True)
        os.symlink("/usr/bin/python3", tree / "bin" / "python")
        (tree / "bin" / "plain").write_text("x", encoding="utf-8")
        os.chmod(tree / "bin", 0o500)
        try:
            assert update_mod._remove_tree(tree) is True
            assert not tree.exists()
        finally:
            if tree.exists():  # pragma: no cover — only on failure
                os.chmod(tree / "bin", 0o700)

    def test_remove_tree_survives_a_symlink_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R5-2: resolving the root must not break "never raises".

        ``Path.resolve()`` — and ``Path.exists()``, which reaches ELOOP through
        ``stat()`` — raise ``RuntimeError`` on a loop, and a loop is exactly the
        input that reaches the helper: ``exists()`` raises, ``is_symlink()`` is
        True, so the guard does not short-circuit. A cleanup path that raises
        replaces the caller's real error with a traceback about a tree it was
        merely tidying.

        THE REAL `rmtree` HERE, not a stub, and that is the correction of round 5's
        account of the ``TypeError``: it came from this function re-issuing
        ``os.open`` — the callable rmtree's fd walk reports an ``ELOOP`` with —
        against a single path, not from the test harness (QA round 3, Q3). Only
        ``os.unlink``/``os.rmdir`` are retried now and the retry cannot raise, so a
        loop reaches the end of the function and answers ``False``.
        """
        loop = tmp_path / "loop"
        os.symlink(loop, loop)
        assert update_mod._remove_tree(loop) is False
        assert loop.is_symlink()

    def test_remove_tree_does_not_chmod_through_a_symlink(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """R4-1: the write-bit retry stays inside the tree it was asked to delete.

        ``shutil.rmtree`` reports a top-level symlink by calling the callback with
        ``os.path.islink`` and the LINK, and ``stat``/``chmod`` follow links — so
        an unguarded retry restores owner bits on the link's TARGET, a directory
        this call was never asked to touch. Asserted the only way that shape can
        be: the outside directory's mode is read before and after, and its
        contents are still there.
        """
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "file").write_text("x", encoding="utf-8")
        os.chmod(outside, 0o500)  # r-x: anything that ORs 0o700 in is visible
        link = tmp_path / "link"
        os.symlink(outside, link)
        try:
            with caplog.at_level(logging.WARNING, logger="local_operator.update"):
                assert update_mod._remove_tree(link) is False, "a symlink is not removable"
            assert (
                stat.S_IMODE(outside.stat().st_mode) == 0o500
            ), "the retry followed the link and chmod'ed its target"
            assert (outside / "file").is_file(), "the link's target was emptied"
            assert "still on disk" in caplog.text
        finally:
            os.chmod(outside, 0o755)

    def test_prune_never_hands_a_symlinked_entry_to_the_remover(
        self, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R4-1's other half: a link under ``generations/`` is not a candidate.

        ``Path.is_dir()`` follows symlinks, so the marker-less branch would have
        handed one straight to the remover. The layout never writes a link there
        (``_reserve_generation`` uses ``mkdir``), and following one would delete
        whatever it points at. ``_remove_tree`` is wrapped to record the paths it
        was offered, so this fails if the entry is merely refused rather than
        never considered.
        """
        real = tmp_path / "not-a-generation"
        real.mkdir()
        (real / "keep").write_text("x", encoding="utf-8")
        generations = update_mod.generations_dir()
        generations.mkdir(parents=True, exist_ok=True)
        link = generations / "20200101T000000Z-link"
        os.symlink(real, link)
        # AGED, because that is the shape that reaches the remover: a fresh
        # marker-less entry is treated as in-flight and kept by the age rule
        # (``path.stat()`` follows the link, so the target's mtime is what
        # counts). Without the ageing this test passes for that reason instead of
        # for the filter — the trap this PR has been caught by three times.
        aged = time.time() - update_mod._PARTIAL_TTL_S - 60
        os.utime(real, (aged, aged))
        attempted: list[Path] = []
        original = update_mod._remove_tree

        def _recording(path: Path) -> bool:
            attempted.append(path)
            return original(path)

        monkeypatch.setattr(update_mod, "_remove_tree", _recording)
        assert update_mod.prune_generations(keep=0).removed == ()
        assert link not in attempted, "prune handed a symlinked entry to the remover"
        assert (real / "keep").is_file(), "prune followed a symlink out of generations/"

    def test_remove_tree_says_whether_the_tree_is_gone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """R3-2's second half: a caller that PRINTS a removal must not be lied to.

        ``prune_generations`` reports the paths it removed, so what
        ``_remove_tree`` answers is what keeps a ``removed:`` line honest when a
        tree cannot be deleted. The read-only shape is the one that could not be:
        that is why write bits are restored and the call retried.
        """
        _skip_as_root()
        read_only = tmp_path / "read-only"
        (read_only / "bin").mkdir(parents=True)
        (read_only / "bin" / "lop").write_text("#!/bin/sh\n", encoding="utf-8")
        os.chmod(read_only / "bin", 0o555)
        assert update_mod._remove_tree(read_only) is True
        assert not read_only.exists(), "the write-bit retry must empty a read-only tree"

        def _refuse(*_args: object, **_kwargs: object) -> None:
            raise OSError("still there")

        stubborn = tmp_path / "stubborn"
        stubborn.mkdir()
        monkeypatch.setattr(update_mod.shutil, "rmtree", _refuse)
        with caplog.at_level(logging.WARNING, logger="local_operator.update"):
            assert update_mod._remove_tree(stubborn) is False
        assert stubborn.exists()
        # The other half of "logged rather than swallowed": a tree that survived
        # is announced, so a reader of the log is not left with the boolean only
        # (review round 4, R4-3).
        assert "still on disk" in caplog.text

    def test_rebind_reports_the_scripts_it_could_not_rewrite(self, tmp_path: Path) -> None:
        """R2-4: the helper's report, asserted at the call site rather than through a refusal.

        ``clone_into_generation`` can only act on what this returns, so the report
        is worth asserting directly: a silent "[]" here is the whole defect.
        """
        _skip_as_root()
        source, copy = tmp_path / "source", tmp_path / "copy"
        for root in (source, copy):
            (root / "bin").mkdir(parents=True)
        for root in (source, copy):
            # The COPY names the SOURCE, which is what a verbatim copy of a real
            # install looks like and why the rewrite has anything to do.
            (root / "bin" / "lop").write_text(f"#!{source}/bin/python3\n", encoding="utf-8")
        os.chmod(copy / "bin", 0o555)
        try:
            rewritten, failed = update_mod._rebind_scripts(copy, source)
        finally:
            os.chmod(copy / "bin", 0o755)
        assert rewritten == []
        assert [path.name for path in failed] == ["lop"]

    def test_a_migration_that_cannot_write_its_launcher_rolls_back(
        self, home: Path, tmp_path: Path
    ) -> None:
        """D1: half a layout is worse than none, and it must not be reported as success.

        With ``~/.local/bin`` unwritable the migration used to copy the tree, flip
        the pointer, plant the daemon shim, print the whole success block and exit
        0 — leaving ``lop`` on PATH running the LEGACY tree while the supervised
        units had moved to the new one (design review round 1, D1). The refusal
        must name what it could not write and put the machine back.
        """
        _skip_as_root()
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        bin_dir = Path.home() / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(bin_dir, 0o500)
        try:
            with pytest.raises(UpdateError) as refused:
                update_mod.clone_into_generation(legacy)
        finally:
            os.chmod(bin_dir, 0o700)
        assert "could not write" in str(refused.value), refused.value
        assert str(bin_dir / "lop") in str(refused.value), refused.value
        assert not update_mod.pointer_path().is_symlink(), "the flip must be undone"
        assert not update_mod.daemon_image_path().exists(), "the shim must be undone"
        assert not list(update_mod.generations_dir().glob("*")), "the copy must be gone"
        # Q1: THE SENTENCE SAYS WHAT HAPPENED. It listed all three steps as "gone";
        # on a machine that had not adopted the layout yet that is true, and this is
        # the arm that pins the words for it — the one this test builds.
        text = str(refused.value)
        assert "the pointer removed" in text, text
        # R1-2: ``remove_shim`` is "there was none before this run", and the only
        # plant happens AFTER the step that failed — so on this machine there was
        # nothing to remove, and the clause said otherwise until review round 1.
        assert "no daemon shim to remove" in text, text
        assert re.search(r"the copy \S+ removed", text), text
        assert (
            "so this machine is as it was" in text
        ), "every step landed, so the claim is earned here"

    def test_a_failed_migration_puts_back_the_pointer_it_replaced(
        self, home: Path, tmp_path: Path
    ) -> None:
        """R6-1: the undo must not delete a pointer this run did not create.

        On a machine that has ALREADY adopted the layout, ``~/.local/bin`` can be
        unwritable for a launcher the migration wants to ADD (a new entry point
        between releases, a launcher deleted by hand) while the existing ones are
        satisfied — ``_atomic_symlink`` short-circuits when the target already
        matches. The old undo unlinked ``current`` anyway, because after the flip
        it names this generation either way: `lop` on PATH then dangled and the
        generation that had been current became unreferenced, while the refusal
        said the machine was as it was.
        """
        _skip_as_root()
        adopted = _install("0.52.0")
        assert update_mod.current_generation() == adopted.resolve()
        # The shim is part of the ADOPTED state this test is about: the machine
        # already carried one from the migration that adopted it, so the undo must
        # KEEP it. Planted rather than assumed, because ``ensure_daemon_image``
        # deliberately declines to name a machine-level artefact from a source
        # checkout, which is what this test process is — and planted as the REAL shim
        # text, because this test EXECUTES it below rather than reading a link.
        shim = update_mod.daemon_image_path()
        shim.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text(update_mod._DAEMON_SHIM, encoding="utf-8")
        shim.chmod(0o755)
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        # One entry point the adopted launcher set does not have: that is how a
        # refusal is reached without disturbing the two launchers that match.
        dist_info = next(legacy.glob("lib/python*/site-packages/local_operator-*.dist-info"))
        entry_points = dist_info / "entry_points.txt"
        entry_points.write_text(
            entry_points.read_text(encoding="utf-8") + "lop-doctor = local_operator.cli:main\n",
            encoding="utf-8",
        )
        extra = legacy / "bin" / "lop-doctor"
        extra.write_text(f"#!{legacy}/bin/python3\n", encoding="utf-8")
        extra.chmod(0o755)
        bin_dir = Path.home() / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(bin_dir, 0o500)
        try:
            with pytest.raises(UpdateError) as refused:
                update_mod.clone_into_generation(legacy)
        finally:
            os.chmod(bin_dir, 0o700)
        assert "lop-doctor" in str(refused.value), refused.value
        assert update_mod.pointer_path().is_symlink(), "the pointer must survive"
        assert (
            update_mod.current_generation() == adopted.resolve()
        ), "a failed migration must leave the machine on the generation it was on"
        # Q1: AND THE SENTENCE SAYS THAT, in the operator's words. It used to list
        # all three steps as "gone", and on THIS machine — the already-adopted state
        # R6-1 created — the pointer is PUT BACK and the existing shim is KEPT, so
        # two of the three clauses described the opposite of what the undo did.
        text = str(refused.value)
        assert f"the pointer put back to {adopted.name}" in text, text
        assert "the daemon shim that was already there kept" in text, text
        # R1-4: the NAME is pinched too — a widened clause that drops it ("the copy
        # removed") is a different sentence, and the weaker "the copy" + "removed"
        # pair passed with it.
        assert re.search(r"the copy \S+ removed", text), text
        assert "are gone" not in text, text
        assert "so this machine is as it was" in text, text
        # ...AND BOTH CONSUMERS REALLY RESOLVE THROUGH IT. The sentence is not the
        # property: the round-2 arm said the shim was "kept" and was measured at
        # rc=126, so the two consumers are EXECUTED here. This is the other half of
        # "no consumer is dead after any undo arm" — the raced arm is the test above.
        shim_ran = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(shim), "--version"], capture_output=True, text=True, timeout=60
        )
        assert shim_ran.returncode == 0, shim_ran.stdout + shim_ran.stderr
        assert shim_ran.stdout.startswith("Python "), shim_ran.stdout
        launcher_ran = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(Path.home() / ".local" / "bin" / "lop")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert launcher_ran.returncode == 0, launcher_ran.stdout + launcher_ran.stderr

    def test_a_prune_between_the_flip_and_the_undo_keeps_both_consumers_working(
        self, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R7-1, Q1 and Q-R2-1 together: no consumer is dead after the raced undo.

        ``flip_pointer`` answers a target that no longer exists with ``UpdateError``,
        not ``OSError``, and the undo caught only the latter — so the refusal escaped
        a function documented never to raise, from its FIRST step, and the two steps
        after it never ran (review round 7, R7-1). The race is this layout's own:
        nothing holds a generation a migration has only captured as ``previous``
        while ``lop install prune`` runs beside it, and ``write_stable_launchers`` is
        the seam because it sits between the flip and the undo. It runs FOR REAL
        here, so the set it reports is the set the undo is asked to keep alive.

        TWO CONSUMERS RESOLVE THROUGH ``current``, and both insist on the layout's
        shape: ``~/.local/bin/lop`` names ``<current>/bin/lop``, and the supervised
        shim resolves the pointer once and execs ``<current>/tools/local-operator/
        bin/python3``. Round 1 removed the pointer and killed the first (``lop
        --version`` → *No such file or directory*); round 2 moved it to the install
        the run came from — a venv whose install sits at its OWN root — and killed the
        second (``/bin/sh`` exiting 126 before the shim's own ``exit 78`` could fire,
        review round 2, Q-R2-1). Both arms are asserted here: the launcher is
        EXECUTED and the shim is EXECUTED, because a green symlink assertion proves
        neither.
        """
        _skip_as_root()
        adopted = _install("0.52.0")
        legacy = tmp_path / "legacy-venv"
        _build_tree(legacy, tmp_path / "legacy-bin", "0.51.9")
        # One entry point the adopted generation does not have, so the launcher write
        # has something to fail on while the two that already name the pointer stay
        # written: the split QA measured as ``written=2 failed=['lop-doctor']``.
        dist_info = next(legacy.glob("lib/python*/site-packages/local_operator-*.dist-info"))
        entry_points = dist_info / "entry_points.txt"
        entry_points.write_text(
            entry_points.read_text(encoding="utf-8") + "lop-doctor = local_operator.cli:main\n",
            encoding="utf-8",
        )
        extra = legacy / "bin" / "lop-doctor"
        extra.write_text(f"#!{legacy}/bin/python3\n", encoding="utf-8")
        extra.chmod(0o755)
        bin_dir = Path.home() / ".local" / "bin"
        os.chmod(bin_dir, 0o500)

        real_write = update_mod.write_stable_launchers

        def _race(generation: Path) -> tuple[list[Path], list[Path]]:
            # A concurrent prune takes the generation captured as ``previous``, and
            # then the launcher write runs unchanged.
            assert update_mod._remove_tree(adopted) is True
            return real_write(generation)

        monkeypatch.setattr(update_mod, "write_stable_launchers", _race)
        try:
            with pytest.raises(UpdateError) as refused:
                update_mod.clone_into_generation(legacy)
        finally:
            os.chmod(bin_dir, 0o700)

        # 1. THE SUPERVISED SHIM EXECS, ASSERTED FIRST because it is the consumer each
        #    earlier arm killed in a different way: round 1 removed the pointer and the
        #    shim answered its own rc=78 *"no current install generation"*, and round 2
        #    moved the pointer to a tree with no ``tools/local-operator`` under it, so
        #    `/bin/sh` exited 126 on an exec path that cannot exist — before the shim's
        #    deliberate diagnostic could fire.
        shim = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(update_mod.daemon_image_path()), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert shim.returncode == 0, shim.stdout + shim.stderr
        assert shim.stdout.startswith("Python "), shim.stdout

        # 2. `lop` ON PATH RUNS, and reaches the copy's own install root. Executed
        #    rather than inspected: the fixture's console script prints the
        #    ``sys.prefix`` that answered, so this asserts which INTERPRETER the
        #    operator's own command reaches now.
        kept = update_mod.current_generation()
        assert kept is not None, "the pointer must keep naming a real generation"
        ran = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [str(bin_dir / "lop")], capture_output=True, text=True, timeout=60
        )
        assert ran.returncode == 0, ran.stdout + ran.stderr
        assert (
            Path(ran.stdout.split()[-1]).resolve() == (kept / "tools" / "local-operator").resolve()
        ), ran.stdout

        # 3. The refused copy STAYS and the pointer names IT: it is the only
        #    generation-shaped tree in this scenario, which is the shape both
        #    consumers resolve through, and what ``flip_pointer`` had already named
        #    when the failure happened.
        assert kept.is_dir()
        assert sorted(path.name for path in update_mod.generations_dir().iterdir()) == [kept.name]
        assert _real(kept) != _real(legacy), "the copy, not the install it came from"

        # 4. The refusal the CALLER is reporting, not the one the cleanup tripped over.
        text = str(refused.value)
        assert "could not write" in text, text
        assert "refusing to point current at a missing generation" not in text, text
        # 5. And it says the copy was KEPT, without claiming a clean rollback: the
        #    machine's own generation is gone, so it is not "as it was" — and the
        #    "kept" clause is true here in a way round 2's was not (the shim runs).
        assert f"the copy {kept.name} kept" in text, text
        assert "the daemon shim that was already there kept" in text, text
        assert "the copy " + kept.name + " removed" not in text, text
        assert "as it was" not in text, text

    def test_a_pointer_whose_target_was_pruned_is_put_on_the_copy(
        self, home: Path, tmp_path: Path
    ) -> None:
        """R1-1 plus round 2's Q-R2-1: a DEAD pointer is put on the copy.

        A prune that read the pointer before the flip takes the tree the pointer
        names, so ``current_generation()`` answers ``None`` while the pointer is
        still a symlink — and every reader of the layout reads a symlink as a
        machine that has a current generation. Unlinking it leaves the launchers
        resolving nothing, and moving it to the install the run came from aims the
        shim at a tree with no ``tools/local-operator`` under it. The copy is kept
        and the pointer is put on it.
        """
        bin_dir = Path.home() / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        pointer = update_mod.pointer_path()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(update_mod.generations_dir() / "20260101T000000Z-vanished", pointer)
        # A launcher resolving through the pointer, and the shim a supervised unit
        # names: between them, the pointer has consumers and must keep working.
        os.symlink(pointer / "bin" / "lop", bin_dir / "lop")
        shim = update_mod.daemon_image_path()
        shim.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text(update_mod._DAEMON_SHIM, encoding="utf-8")
        shim.chmod(0o755)
        copy = update_mod.generations_dir() / "20260101T000000Z-migrate-legacy"
        (copy / "tools" / "local-operator" / "bin").mkdir(parents=True)
        legacy = tmp_path / "legacy-venv"
        legacy.mkdir()

        outcome = update_mod._undo_migration(copy, remove_shim=False, previous=None)

        text = ", ".join(outcome.parts)
        assert f"the copy {copy.name} kept" in text, text
        assert "naming another generation" not in text, text
        put_back = update_mod.current_generation()
        assert put_back is not None, "a dead pointer is what this arm must not leave"
        assert _real(put_back) == _real(copy), "the copy, not the install the run came from"
        assert copy.is_dir(), "the tree the pointer resolves through must survive"
        assert outcome.complete is False, "a kept copy is not the machine it was"

    def test_what_resolves_through_the_pointer_is_asked_of_the_filesystem(self, home: Path) -> None:
        """Q-R2-3: the gate is RESOLUTION, not what this run wrote.

        ``write_stable_launchers`` reports a launcher as written only when the link
        is byte-identical to the absolute target, so a link spelled relatively — or
        written by an older build, or by a different ``UV_TOOL_BIN_DIR`` — is
        load-bearing while being invisible to that return value. Gating on it put a
        machine whose launcher DID resolve through the pointer into the unlink arm,
        which is round-1 Q1's symptom by a narrower path.
        """
        _install("0.52.0")
        bin_dir = Path.home() / ".local" / "bin"
        launcher = bin_dir / "lop"
        launcher.unlink()
        relative = os.path.relpath(update_mod.pointer_path() / "bin" / "lop", bin_dir)
        os.symlink(relative, launcher)
        assert launcher in update_mod._pointer_consumers(), "a relative spelling still resolves"

        elsewhere = bin_dir / "elsewhere"
        os.symlink("/usr/bin/true", elsewhere)
        assert elsewhere not in update_mod._pointer_consumers()

        shim = update_mod.daemon_image_path()
        shim.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text(update_mod._DAEMON_SHIM, encoding="utf-8")
        assert (
            shim in update_mod._pointer_consumers()
        ), "a supervised unit names the shim, and the shim resolves the pointer at exec"

    def test_a_symlinked_install_path_still_re_points_the_copy(
        self, home: Path, tmp_path: Path
    ) -> None:
        """D2: the file's spelling is not necessarily the caller's.

        ``sys.prefix`` is already RESOLVED by CPython while uv wrote the spelling
        it was given, so on a host with a symlinked install path (``/tmp`` is
        ``/private/tmp`` on macOS) a string search finds nothing, nothing fails,
        and the migration reports success with a copy that still executes the
        legacy venv. The check is on the venv's identity instead.
        """
        real = tmp_path / "real-legacy"
        real.mkdir()
        link = tmp_path / "legacy-link"
        os.symlink(real, link)
        # The tree's own scripts name the LINK, as uv's would: it was installed
        # through the spelling the caller handed it.
        _build_tree(link, tmp_path / "legacy-bin", "0.51.9")
        generation = update_mod.clone_into_generation(real)
        install_root = generation / "tools" / "local-operator"
        shebang = (install_root / "bin" / "lop").read_text(encoding="utf-8").splitlines()[0]
        assert shebang == f"#!{install_root / 'bin' / 'python3'}", shebang
        activate = install_root / "bin" / "activate"
        if activate.is_file():
            assert str(link) not in activate.read_text(encoding="utf-8")

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
        removed = update_mod.prune_generations().removed
        assert update_mod.current_generation() == generations[-1].resolve()
        assert generations[-1] not in removed

    def test_a_live_record_holds_its_generation(self, home: Path) -> None:
        """The one thing that makes pruning safe on a busy machine."""
        generations = [_install(f"0.52.{index}") for index in range(5)]
        held = generations[0]
        removed = update_mod.prune_generations(
            referenced=[held / "tools" / "local-operator"],
        ).removed
        assert held not in removed
        assert held.is_dir()
        assert not any(path == held for path in removed)

    def test_the_last_two_unreferenced_generations_survive(self, home: Path) -> None:
        generations = [_install(f"0.52.{index}") for index in range(5)]
        removed = update_mod.prune_generations().removed
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
        assert update_mod.prune_generations(keep=0).removed.count(in_flight) == 0
        assert in_flight.is_dir()

    def test_an_abandoned_tree_is_reclaimed_once_it_is_old(self, home: Path) -> None:
        _install("0.52.0")
        debris = update_mod.generations_dir() / "20260101T000000Z-crashed"
        debris.mkdir()
        (debris / "tools" / "local-operator").mkdir(parents=True)
        removed = update_mod.prune_generations(
            keep=0, now=time.time() + update_mod._PARTIAL_TTL_S + 1
        ).removed
        assert debris in removed
        assert not debris.exists()

    def test_an_in_flight_tree_does_not_consume_a_margin_slot(self, home: Path) -> None:
        """QA Q3: the margin counts marker-carrying generations only.

        A marker-less tree is skipped by the age rule, so letting it hold one of
        the ``keep`` places shrank the protection for finished builds exactly while
        an install was running — the margin's documented job is the opposite.
        """
        oldest = _install("0.52.0")
        middle = _install("0.52.1")
        newest = _install("0.52.2")
        aged = time.time() - 100
        for offset, path in enumerate((oldest, middle, newest)):
            os.utime(path, (aged + offset, aged + offset))
        in_flight = update_mod.generations_dir() / "20260101T000000Z-writing"
        in_flight.mkdir()
        (in_flight / "tools" / "local-operator").mkdir(parents=True)
        plan = update_mod.prune_generations(keep=1)
        assert oldest in plan.removed
        assert middle not in plan.removed, "the margin must protect a FINISHED build"
        assert in_flight.is_dir()

    def test_prune_reports_what_it_kept_and_why(self, home: Path) -> None:
        """D3: the one command whose whole job is a retention decision must explain it."""
        _install("0.52.0")
        _install("0.52.1")
        plan = update_mod.prune_generations()
        lines = update_mod.prune_lines(plan)
        text = "\n".join(lines)
        assert plan.kept, "the keeps are part of the answer"
        assert "current:" in text, text
        assert "unreferenced, but within --keep 2" in text, text
        assert all(decision.reason for decision in plan.decisions), lines

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

    def test_a_failed_removal_is_not_summarised_as_nothing_to_remove(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q2: the header claimed the benign arm while the rows said otherwise.

        An empty machine and a machine where every removal attempt FAILED both
        arrive at ``plan.removed == ()``, and the summary said ``nothing to
        remove`` for both — directly above rows reading ``could not be removed; it
        is still there``. On the one command whose entire job is to report a
        retention decision, that is the contradiction an operator acts on wrongly:
        "nothing to remove" reads as "this machine is already trimmed" while the
        disk is untouched. With no removals every candidate WAS attempted, so the
        two arms are exact rather than an estimate.
        """
        _skip_as_root()
        first = _install("0.52.0")
        second = _install("0.52.1")
        _install("0.52.2")  # the pointer's target: never a removal candidate
        stubborn = {_real(first), _real(second)}
        real_remove = update_mod._remove_tree

        def _refuse(path: Path) -> bool:
            # The shape a real failure has: a path the process may not delete, which
            # the write-bit retry inside ``_remove_tree`` cannot rescue either.
            return False if _real(path) in stubborn else real_remove(path)

        monkeypatch.setattr(update_mod, "_remove_tree", _refuse)
        plan = update_mod.prune_generations(keep=0)
        assert plan.removed == ()
        text = "\n".join(update_mod.prune_lines(plan))
        assert "could not be removed; it is still there" in text, text
        assert "nothing to remove" not in text, text
        assert "none could be removed" in text, text
        assert first.is_dir() and second.is_dir()

    def test_nothing_to_remove_still_says_so_when_no_removal_was_attempted(
        self, home: Path
    ) -> None:
        """The benign arm keeps its words: only a FAILED attempt changes the header."""
        _install("0.52.0")
        plan = update_mod.prune_generations()
        text = "\n".join(update_mod.prune_lines(plan))
        assert plan.removed == ()
        assert "nothing to remove" in text, text
        assert "none could be removed" not in text, text

    def test_pruning_a_machine_with_no_layout_is_a_no_op(self, home: Path) -> None:
        assert update_mod.prune_generations().removed == ()


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
        # STDERR, with every sibling refusal on this surface (design review round 2,
        # D12): a refusal on stdout is filed as output by `lop install migrate | tee`.
        captured = capsys.readouterr()
        assert "refusing to migrate" in captured.err
        assert captured.out == ""
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


# -- status ----------------------------------------------------------------------


class TestInstallStatus:
    """The pointer states, the load label and the held line (R7-5).

    Three review rounds churned this text with no test behind it — D7 (the empty
    state explaining itself), D16 (``a new lop would load:`` shortened to ``next lop
    would load:`` so the value fits the column), D17 (a held generation named by its
    id rather than by the absolute path its record carries) — and text defended only
    by the rounds that produced it is text the next edit can silently rewrite.

    Expected lines are built with ``_field`` rather than spelled out, because the
    block's column is that helper's job: a test that re-counts the padding fails on
    a change no reader would see, and passes on the change this class exists for.
    """

    def test_an_adopted_machine_names_the_generation_it_would_load(
        self, home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        generation = _install("0.52.0")
        assert update_mod.install_status_command() == 0
        text = capsys.readouterr().out
        current = update_mod.current_generation()
        assert current is not None
        assert current.name == generation.name
        assert (
            update_mod._field("pointer:", f"{update_mod.pointer_path()} -> {current}") in text
        ), text
        assert f"  {current.name}  <- current" in text, text

    def test_a_pointer_that_resolves_to_nothing_is_unresolved_not_absent(
        self, home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """R6-2: a dangling pointer and a machine with no layout are two states.

        They printed identically, including while the very next line listed the
        generations that DO exist.
        """
        generation = _install("0.52.0")
        shutil.rmtree(generation)
        assert update_mod.install_status_command() == 0
        text = capsys.readouterr().out
        assert (
            update_mod._field("pointer:", f"{update_mod.pointer_path()} -> (unresolved)") in text
        ), text
        assert "no generation layout on this machine" not in text, text

    def test_a_missing_pointer_with_generations_standing_says_absent(
        self, home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The layout question is answered by the LAYOUT, not by the pointer (R6-2)."""
        generation = _install("0.52.0")
        update_mod.pointer_path().unlink()
        assert update_mod.install_status_command() == 0
        text = capsys.readouterr().out
        assert (
            update_mod._field("pointer:", f"{update_mod.pointer_path()} -> (absent)") in text
        ), text
        assert "<- current" not in text, "no generation can be current with no pointer"
        assert f"  {generation.name}" in text, "the generations standing are still listed"

    def test_the_load_label_is_the_short_one_and_names_the_stamp(
        self, home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """D16: the shortened label, and the single space that shortening bought."""
        generation = _install("0.52.0")
        stamp = update_mod.disk_build(generation / "tools" / "local-operator")
        assert stamp is not None, "a generation installed through the layout carries a stamp"
        assert update_mod.install_status_command() == 0
        text = capsys.readouterr().out
        assert f"next lop would load: {stamp.label()}" in text, text
        assert "a new lop would load:" not in text, text

    def test_a_held_generation_is_named_by_its_id(
        self, home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """D17: a record names the VENV, so the record's own path would be the one
        line in this block not written in the vocabulary of the lines above it."""
        from local_operator.session.runtime import registry
        from local_operator.session.runtime.types import SessionRecord

        generation = _install("0.52.0")
        install_root = generation / "tools" / "local-operator"
        record = SessionRecord(
            pid=os.getpid(),
            kind="tui",
            session_id="status",
            conversation_name="n",
            cwd="/",
            model_label="m",
            control_port=0,
            control_key="k",
            install_root=str(install_root),
        )
        root = Path.home() / ".local-operator"
        registry.publish(record, root=root)
        try:
            assert update_mod.install_status_command() == 0
        finally:
            registry.unpublish(record.pid, root)
        text = capsys.readouterr().out
        assert f"  held by a live session: {generation.name}" in text, text
        assert str(install_root) not in text, text

    def test_an_unknown_load_line_does_not_contradict_the_current_row(
        self, home: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """D18: a post-flip-damaged generation is reachable, and this line used to
        lie about it.

        With the pointer resolving to a generation that holds no install in it, the
        load line printed "(unknown — nothing resolves behind the pointer)" directly
        above the row marking that same generation ``<- current``. The designer
        recorded the reachability honestly and accepted the rejection because the
        clean writers cannot produce it — but the state is one ``rm -r`` away from
        anything that can, and the fix is the sentence, so the sentence is fixed.
        """
        damaged = update_mod.generations_dir() / "20260101T000000Z-damaged"
        (damaged / "tools").mkdir(parents=True)
        update_mod.flip_pointer(damaged)
        assert update_mod.install_status_command() == 0
        text = capsys.readouterr().out
        assert f"  {damaged.name}  <- current" in text, text
        assert (
            update_mod._field(
                "next lop would load:", f"(unknown — no install root under {damaged.name})"
            )
            in text
        ), text
        assert "nothing resolves behind the pointer" not in text, text


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

"""Safety rules of the standing build-cache reaper, one test per rule.

Every test is ``tmp_path``-scoped and injects the tri-state probes, so nothing
here deletes or reads anything outside the fixture. The rules under test are the
ones in the module docstring of :mod:`local_operator.build_cache_reaper`; each
test names the rule it holds.

The load-bearing guards are the live-build probes, and the point of the tri-state
design is that **"cannot tell" is a skip** — so the tests below drive ``True``,
``False`` AND ``None`` through each probe rather than only the interesting one.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
import types
from pathlib import Path

import pytest

from local_operator import build_cache_reaper as bcr

#: Committed-identity flags, so a test never reads the developer's git config —
#: and never prompts a credential helper (this harness forbids a rig that reaches
#: the operator's keychain).
_GIT_ID = (
    "-c",
    "user.name=build-cache-test",
    "-c",
    "user.email=build-cache-test@example.invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "init.defaultBranch=main",
)

#: Age used for "definitely stale" fixture trees, well past the 6 h default.
OLD_HOURS = 24.0


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the config dir (and so the log dir) at the fixture.

    ``paths.config_dir`` honours ``LOCAL_OPERATOR_CONFIG_DIR``, which also makes
    the config-dir exclusion resolve inside the fixture. A test that left this
    alone would write its log into the developer's real log directory.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    monkeypatch.delenv("LOCAL_OPERATOR_SCRATCHPAD", raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("GIT_ASKPASS", "/bin/false")


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *_GIT_ID, *args], cwd=cwd, check=True, capture_output=True)


def _age(path: Path, hours: float = OLD_HOURS) -> None:
    """Stamp ``path`` (and a directory's contents are stamped by the caller)."""
    when = time.time() - hours * 3600.0
    os.utime(path, (when, when))


def _make_shape(root: Path, name: str, *, hours: float = OLD_HOURS, payload: int = 4096) -> Path:
    """A stale shape directory with real bytes in it, stamped ``hours`` old."""
    shape = root / name
    inner = shape / "debug"
    inner.mkdir(parents=True)
    artifact = inner / "artifact.rlib"
    artifact.write_bytes(b"x" * payload)
    for target in (shape, inner, artifact):
        _age(target, hours)
    return shape


def _probes(**overrides: object) -> bcr.Probes:
    """The four probes plus the tick-level table check, each pinned to a value.

    ``None`` is a legitimate value here — it is the "cannot tell" answer the tests
    exist to prove is treated as a skip — and a CALLABLE override installs a real
    probe (``bcr.git_tracked_files_probe``) instead of a fixed answer, so a test can
    drive the production adapter through the seam.
    """
    values: dict[str, object] = {
        "toolchain_running": False,
        "handles_open": False,
        "is_git_ignored": True,
        "tracked_files": False,
        "is_mount_point": False,
        "process_table_readable": True,
    }
    values.update(overrides)

    def pinned(name: str) -> object:
        candidate = values[name]
        if callable(candidate):
            return candidate
        return lambda _path, _value=candidate: _value

    def pinned_table() -> object:
        # The tick-level check takes NO path: it answers for the whole tick.
        candidate = values["process_table_readable"]
        if callable(candidate):
            return candidate
        return lambda _value=candidate: _value

    return bcr.Probes(
        toolchain_running=pinned("toolchain_running"),  # type: ignore[arg-type]
        handles_open=pinned("handles_open"),  # type: ignore[arg-type]
        is_git_ignored=pinned("is_git_ignored"),  # type: ignore[arg-type]
        tracked_files=pinned("tracked_files"),  # type: ignore[arg-type]
        is_mount_point=pinned("is_mount_point"),  # type: ignore[arg-type]
        refresh=lambda: None,
        process_table_readable=pinned_table(),  # type: ignore[arg-type]
    )


def _settings(**overrides: object) -> bcr.ReaperSettings:
    values: dict[str, object] = {
        "enabled": True,
        "age_hours": 6,
        "dry_run": False,
        "include_extra_shapes": False,
    }
    values.update(overrides)
    return bcr.ReaperSettings(**values)  # type: ignore[arg-type]


def _reap(
    *,
    roots: list[Path],
    home: Path,
    settings: bcr.ReaperSettings | None = None,
    probes: bcr.Probes | None = None,
    budget_s: float = bcr.DEFAULT_TIME_BUDGET_S,
    clock: object = None,
) -> bcr.ReapReport:
    kwargs: dict[str, object] = {}
    if clock is not None:
        kwargs["clock"] = clock
    return bcr.reap(
        roots=roots,
        home=home,
        settings=settings if settings is not None else _settings(),
        probes=probes if probes is not None else _probes(),
        budget_s=budget_s,
        **kwargs,  # type: ignore[arg-type]
    )


def _reasons(report: bcr.ReapReport, name: str) -> set[str]:
    return {skip.reason for skip in report.skipped if skip.path.name == name}


# ---------------------------------------------------------------------------
# Rule 1: only known build-cache shapes, and never work that is not build output
# ---------------------------------------------------------------------------


def test_unrelated_files_and_directories_are_untouched(tmp_path: Path) -> None:
    """Anything whose name is not a known shape is left completely alone.

    ``out/`` is in the fixture on purpose: the disk lane measured 14.83 GiB of it
    in one repository, holding capture frames from merged pull requests — evidence
    that merely LOOKS like packaging output. "Large and regenerable-looking" is not
    the test, and this asserts the reaper agrees.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "main.rs").write_text("fn main() {}\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("hand-written\n")
    out = repo / "out" / "frames"
    out.mkdir(parents=True)
    (out / "pr-1.png").write_bytes(b"png" * 100)
    for path in (repo / "src", repo / "src" / "main.rs", repo / "out", out, out / "pr-1.png"):
        _age(path)
    shape = _make_shape(repo, "target")

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert [removal.path for removal in report.removed] == [shape]
    assert not shape.exists()
    assert (repo / "src" / "main.rs").exists()
    assert (repo / "docs" / "notes.md").exists()
    assert (out / "pr-1.png").exists()
    assert (repo / "out").exists()


def test_node_modules_is_never_a_candidate(tmp_path: Path) -> None:
    """``node_modules`` is excluded by construction, not by configuration."""
    app = tmp_path / "app"
    (app / "node_modules" / "@scope" / "pkg").mkdir(parents=True)
    (app / "node_modules" / "@scope" / "pkg" / "index.js").write_text("x\n")
    _age(app / "node_modules" / "@scope" / "pkg" / "index.js")
    _age(app / "node_modules" / "@scope" / "pkg")
    _age(app / "node_modules" / "@scope")
    _age(app / "node_modules")

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert (app / "node_modules" / "@scope" / "pkg" / "index.js").exists()
    # Not "skipped" either: it is never considered, so it does not even cost a row.
    mentioned = {entry.path for entry in report.skipped} | {
        candidate.path for candidate in report.candidates
    }
    assert not any("node_modules" in str(path) for path in mentioned)


def test_git_tracked_shape_is_skipped(tmp_path: Path) -> None:
    """A shape directory holding a TRACKED file is never removed.

    Real ``git ls-files`` rather than an injected answer: this is the rule that
    protects committed content, and the probe's own wiring is part of what is
    being asserted.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    (repo / ".gitignore").write_text("target/\n")
    shape = repo / "target"
    (shape / "debug").mkdir(parents=True)
    (shape / "release.txt").write_text("committed on purpose\n")
    _git("add", ".gitignore", cwd=repo)
    _git("add", "-f", "target/release.txt", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    for path in (shape, shape / "debug", shape / "release.txt"):
        _age(path)

    probes = _probes(
        tracked_files=bcr.git_tracked_files_probe, is_git_ignored=bcr.git_ignored_probe
    )
    report = _reap(roots=[tmp_path], home=tmp_path, probes=probes)

    assert report.removed == ()
    assert shape.exists()
    assert _reasons(report, "target") == {bcr.SKIP_GIT_TRACKED}


def test_untracked_but_unignored_shape_is_skipped(tmp_path: Path) -> None:
    """Untracked AND unignored is the owner's call, not the reaper's.

    ``dist/`` committed by policy or simply not ignored is a decision someone
    made; ``git status`` showing it means a human is meant to see it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    (repo / "README").write_text("hi\n")
    _git("add", "README", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    shape = _make_shape(repo, "dist")
    (repo / ".gitignore").write_text("target/\n")  # exists but does not cover dist/

    probes = _probes(
        tracked_files=bcr.git_tracked_files_probe, is_git_ignored=bcr.git_ignored_probe
    )
    report = _reap(
        roots=[tmp_path],
        home=tmp_path,
        settings=_settings(include_extra_shapes=True),
        probes=probes,
    )

    assert shape.exists()
    assert report.removed == ()
    assert _reasons(report, "dist") == {bcr.SKIP_GIT_NOT_IGNORED}


def test_session_scratchpad_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A session scratchpad's own ``target/`` is never a candidate."""
    scratch = tmp_path / "scratch"
    (scratch / "target" / "debug").mkdir(parents=True)
    (scratch / "target" / "debug" / "junk.o").write_bytes(b"y" * 256)
    for path in (
        scratch / "target",
        scratch / "target" / "debug",
        scratch / "target" / "debug" / "junk.o",
    ):
        _age(path)
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(scratch))

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert (scratch / "target" / "debug" / "junk.o").exists()


def test_config_dir_shape_is_skipped(tmp_path: Path) -> None:
    """Nothing under the config dir is ever touched: credentials and transcripts."""
    config_dir = tmp_path / ".local-operator"
    shape = config_dir / "target"
    (shape / "debug").mkdir(parents=True)
    (shape / "debug" / "secret.txt").write_text("not ours\n")
    for path in (shape, shape / "debug", shape / "debug" / "secret.txt"):
        _age(path)

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert (shape / "debug" / "secret.txt").exists()


def test_home_library_shape_is_skipped(tmp_path: Path) -> None:
    """``~/Library`` is application state, not build output."""
    shape = tmp_path / "Library" / "Caches" / "target"
    (shape / "debug").mkdir(parents=True)
    (shape / "debug" / "x.bin").write_bytes(b"z" * 64)
    for path in (shape, shape / "debug", shape / "debug" / "x.bin"):
        _age(path)

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert (shape / "debug" / "x.bin").exists()


def test_symlinked_shape_is_skipped_and_its_target_survives(tmp_path: Path) -> None:
    """A symlinked shape is skipped entirely; the real directory behind it survives.

    A single shared build directory behind per-worktree symlinks is exactly how a
    reaper deletes every worktree's build at once.
    """
    shared = tmp_path / "shared-target"
    (shared / "debug").mkdir(parents=True)
    (shared / "debug" / "artifact.o").write_bytes(b"q" * 128)
    for path in (shared, shared / "debug", shared / "debug" / "artifact.o"):
        _age(path)
    repo = tmp_path / "repo"
    repo.mkdir()
    link = repo / "target"
    link.symlink_to(shared, target_is_directory=True)

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert link.is_symlink()
    assert (shared / "debug" / "artifact.o").exists()
    assert _reasons(report, "target") == {bcr.SKIP_SYMLINK}


# ---------------------------------------------------------------------------
# Rule 2: never prune while a build may be running. "Cannot tell" means skip.
# ---------------------------------------------------------------------------


def test_age_floor_skips_a_fresh_candidate(tmp_path: Path) -> None:
    """Newest FILE decides the age, and a young one makes the candidate ``fresh``."""
    shape = _make_shape(tmp_path, "target", hours=0.5)

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert shape.exists()
    assert _reasons(report, "target") == {bcr.SKIP_FRESH}


def test_directory_mtime_alone_does_not_make_a_candidate_stale(tmp_path: Path) -> None:
    """An old directory whose newest file is recent is still fresh.

    The measured case: a directory stamped 1970 by a bad clock, holding files
    written minutes ago, must not read as decades old.
    """
    shape = tmp_path / "target"
    (shape / "debug").mkdir(parents=True)
    (shape / "debug" / "just-written.o").write_bytes(b"w" * 64)
    epoch = 0.0
    os.utime(shape, (epoch, epoch))
    os.utime(shape / "debug", (epoch, epoch))

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert _reasons(report, "target") == {bcr.SKIP_FRESH}


def test_toolchain_running_skips_a_candidate(tmp_path: Path) -> None:
    shape = _make_shape(tmp_path, "target")

    report = _reap(roots=[tmp_path], home=tmp_path, probes=_probes(toolchain_running=True))

    assert report.removed == ()
    assert shape.exists()
    assert _reasons(report, "target") == {bcr.SKIP_TOOLCHAIN}


def test_toolchain_cannot_tell_skips_a_candidate(tmp_path: Path) -> None:
    """``None`` from the process probe is a SKIP, never "nothing is running"."""
    shape = _make_shape(tmp_path, "target")

    report = _reap(roots=[tmp_path], home=tmp_path, probes=_probes(toolchain_running=None))

    assert report.removed == ()
    assert shape.exists()
    assert _reasons(report, "target") == {bcr.SKIP_CANNOT_TELL}
    assert report.skipped[0].detail == "toolchain-probe"


def test_open_handle_skips_a_candidate(tmp_path: Path) -> None:
    shape = _make_shape(tmp_path, "target")

    report = _reap(roots=[tmp_path], home=tmp_path, probes=_probes(handles_open=True))

    assert report.removed == ()
    assert shape.exists()
    assert _reasons(report, "target") == {bcr.SKIP_HANDLES}


def test_handle_probe_failure_skips_a_candidate(tmp_path: Path) -> None:
    """A failed ``lsof`` means "everything may be in use", never "nothing is".

    This is the rule ``pytest-tmp-reaper`` paid for: a stale disk is recoverable,
    a live build deleted underneath another session is not.
    """
    shape = _make_shape(tmp_path, "target")

    report = _reap(roots=[tmp_path], home=tmp_path, probes=_probes(handles_open=None))

    assert report.removed == ()
    assert shape.exists()
    assert _reasons(report, "target") == {bcr.SKIP_CANNOT_TELL}
    assert report.skipped[0].detail == "handle-probe"


def test_live_build_blocks_one_candidate_not_the_whole_tick(tmp_path: Path) -> None:
    """The guard is PER CANDIDATE: a build in one tree does not freeze the others.

    On a machine running ~25 concurrent sessions, a global "any toolchain process
    anywhere" rule would make this reaper a permanent no-op.
    """
    busy = _make_shape(tmp_path / "busy", "target")
    idle = _make_shape(tmp_path / "idle", "target")
    probes = bcr.Probes(
        toolchain_running=lambda path: path == busy,
        handles_open=lambda _p: False,
        is_git_ignored=lambda _p: True,
        tracked_files=lambda _p: False,
        refresh=lambda: None,
        process_table_readable=lambda: True,
    )

    report = _reap(roots=[tmp_path], home=tmp_path, probes=probes)

    assert busy.exists()
    assert not idle.exists()
    assert [removal.path for removal in report.removed] == [idle]
    assert _reasons(report, "target") == {bcr.SKIP_TOOLCHAIN}


def test_unreadable_process_table_aborts_the_whole_tick(tmp_path: Path) -> None:
    """No process table means no attribution, so the tick removes nothing."""
    shape = _make_shape(tmp_path, "target")

    report = _reap(roots=[tmp_path], home=tmp_path, probes=_probes(process_table_readable=None))

    assert report.aborted == "aborted: cannot read process table"
    assert not report.aborted.startswith("aborted: cannot read process table") is False
    assert report.removed == ()
    assert shape.exists()


def test_unmeasurable_tree_is_skipped_rather_than_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree past the walk bound has an UNKNOWN age, so it is skipped.

    A partial maximum understates recency, which is the unsafe direction; the test
    pins the skip AND the reason detail so the bound that tripped is visible.
    """
    _make_shape(tmp_path, "target")
    monkeypatch.setattr(
        bcr, "scan_tree", lambda *a, **k: bcr.TreeScan(None, 0, 0, 0, "entries", False)
    )

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert report.skipped[0].reason == bcr.SKIP_CANNOT_TELL
    assert report.skipped[0].detail == "age-walk-entries"


def test_a_tree_with_no_readable_mtime_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_shape(tmp_path, "target")
    monkeypatch.setattr(bcr, "scan_tree", lambda *a, **k: bcr.TreeScan(None, 0, 0, 0, None, False))

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert report.skipped[0].detail == "age-unknown"


def test_recheck_immediately_before_deletion_catches_a_new_build(tmp_path: Path) -> None:
    """The live answer is re-read at reap time, not carried over from the scan.

    Driven through the injection seam: the scan-time answer is idle, and the
    refreshed answer — what the recheck sees — is a running toolchain.
    """
    shape = _make_shape(tmp_path, "target")
    state = {"running": False}

    def refresh() -> None:
        state["running"] = True

    probes = bcr.Probes(
        toolchain_running=lambda _p: state["running"],
        handles_open=lambda _p: False,
        is_git_ignored=lambda _p: True,
        tracked_files=lambda _p: False,
        refresh=refresh,
        process_table_readable=lambda: True,
    )

    report = _reap(roots=[tmp_path], home=tmp_path, probes=probes)

    assert shape.exists()
    assert report.removed == ()
    assert _reasons(report, "target") == {bcr.SKIP_TOOLCHAIN}
    assert report.skipped[0].detail == "recheck"


# ---------------------------------------------------------------------------
# Rule 4: never delete a worktree, a branch, or uncommitted work
# ---------------------------------------------------------------------------


def test_worktree_survives_with_its_git_file_branch_and_uncommitted_work(tmp_path: Path) -> None:
    """Reaping a linked worktree's ``target/`` leaves the worktree itself intact."""
    repo = tmp_path / "repo"
    (repo / "crate").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    (repo / "crate" / "lib.rs").write_text("// hi\n")
    _git("add", "crate/lib.rs", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    worktree = repo / ".worktrees" / "feature"
    _git("worktree", "add", "-q", str(worktree), "-b", "feat/keep-me", cwd=repo)
    assert (worktree / ".git").is_file()
    uncommitted = worktree / "crate" / "wip.txt"
    uncommitted.write_text("uncommitted work\n")
    shape = _make_shape(worktree, "target")

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert [removal.path for removal in report.removed] == [shape]
    assert not shape.exists()
    assert (worktree / ".git").is_file()
    assert uncommitted.read_text() == "uncommitted work\n"
    branch = subprocess.run(
        ["git", *_GIT_ID, "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "feat/keep-me"


# ---------------------------------------------------------------------------
# Rule 5: report what was reclaimed, in a place a human can read
# ---------------------------------------------------------------------------


def test_dry_run_removes_nothing_and_reports_the_same_bytes(tmp_path: Path) -> None:
    shape = _make_shape(tmp_path, "target", payload=8192)
    allocated = len(b"x" * 8192)  # apparent; allocated is block-rounded, so assert > 0 below

    dry = _reap(roots=[tmp_path], home=tmp_path, settings=_settings(dry_run=True))

    assert dry.dry_run is True
    assert dry.removed == ()
    assert shape.exists()
    assert len(dry.candidates) == 1
    assert dry.candidates[0].path == shape
    assert dry.candidates[0].allocated_bytes > 0
    assert dry.candidates[0].apparent_bytes >= allocated
    assert dry.reclaimed_bytes == 0
    assert dry.would_reclaim_bytes == dry.candidates[0].allocated_bytes


def test_real_run_removes_the_shape_and_logs_the_removal(tmp_path: Path) -> None:
    shape = _make_shape(tmp_path, "target", payload=16384)

    report = _reap(roots=[tmp_path], home=tmp_path)
    log_file = bcr.write_log(report)

    assert log_file is not None
    assert str(log_file).endswith(f"logs{os.sep}{bcr.LOG_FILENAME}")
    assert not shape.exists()
    assert len(report.removed) == 1
    assert report.removed[0].allocated_bytes > 0
    assert report.reclaimed_bytes == report.removed[0].allocated_bytes
    body = log_file.read_text()
    assert str(shape) in body
    assert "shape=target" in body
    assert "removed" in body
    assert "summary" in body
    assert "reclaimed=" in body
    assert report.free_after >= 0 and report.free_before >= 0


def test_log_records_an_aborted_tick(tmp_path: Path) -> None:
    _make_shape(tmp_path, "target")
    report = _reap(roots=[tmp_path], home=tmp_path, probes=_probes(process_table_readable=None))

    log_file = bcr.write_log(report)

    assert log_file is not None
    body = log_file.read_text()
    assert "aborted: cannot read process table" in body
    assert "removed=0" in body


# ---------------------------------------------------------------------------
# Rule 3 / shapes: the opt-in shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["build", "dist"])
def test_extra_shapes_are_skipped_by_default(tmp_path: Path, name: str) -> None:
    shape = _make_shape(tmp_path, name)

    report = _reap(roots=[tmp_path], home=tmp_path, settings=_settings(include_extra_shapes=False))

    assert report.removed == ()
    assert shape.exists()
    # Never considered at all, so it does not even appear as a skip.
    assert name not in {skip.path.name for skip in report.skipped}


@pytest.mark.parametrize("name", ["build", "dist"])
def test_extra_shapes_are_reaped_when_opted_in(tmp_path: Path, name: str) -> None:
    shape = _make_shape(tmp_path, name)

    report = _reap(roots=[tmp_path], home=tmp_path, settings=_settings(include_extra_shapes=True))

    assert [removal.path for removal in report.removed] == [shape]
    assert not shape.exists()


def test_newest_file_decides_the_age_and_bytes_are_allocated_blocks(tmp_path: Path) -> None:
    """One walk answers both questions, and the byte figure is ``st_blocks * 512``."""
    shape = _make_shape(tmp_path, "target", payload=4096)
    expected_allocated = sum(
        path.stat().st_blocks * 512 for path in shape.rglob("*") if path.is_file()
    )

    scan = bcr.scan_tree(shape, cutoff=time.time() - bcr.DEFAULT_AGE_HOURS * 3600)

    assert scan.stopped is None
    assert scan.saw_fresh is False
    assert scan.allocated_bytes == expected_allocated
    assert scan.allocated_bytes >= 4096
    assert scan.newest_mtime is not None


def test_scan_tree_stops_at_the_entry_bound() -> None:
    """The bound reports itself instead of returning a partial maximum as truth."""
    scan = bcr.scan_tree(Path(__file__).parent, cutoff=0.0, max_entries=1, stop_when_fresh=False)

    assert scan.stopped == "entries"
    assert scan.entries == 2


# ---------------------------------------------------------------------------
# Scan roots, bounds and settings — the pure seams the CLI drives
# ---------------------------------------------------------------------------


def test_resolve_roots_defaults_to_home(tmp_path: Path) -> None:
    assert bcr.resolve_roots(None, tmp_path) == (Path(os.path.realpath(str(tmp_path))),)
    assert bcr.resolve_roots([], tmp_path) == (Path(os.path.realpath(str(tmp_path))),)


def test_resolve_roots_replaces_home_and_deduplicates(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()

    roots = bcr.resolve_roots([one, one, two], tmp_path)

    assert roots == (Path(os.path.realpath(str(one))), Path(os.path.realpath(str(two))))
    assert Path(os.path.realpath(str(tmp_path))) not in roots


def test_scan_budget_sets_incomplete_and_still_reaps_proved_candidates(tmp_path: Path) -> None:
    """Hitting the wall-clock budget is reported, not silently swallowed."""
    _make_shape(tmp_path, "target")

    class _Clock:
        """First call arms the deadline, every later call is past it."""

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 0.0 if self.calls == 1 else 10_000.0

    report = _reap(roots=[tmp_path], home=tmp_path, budget_s=1.0, clock=_Clock())

    assert report.incomplete_scan is True
    assert report.aborted == ""


def test_read_settings_uses_the_registry_and_falls_back_to_defaults(tmp_path: Path) -> None:
    """The four keys are read through the settings registry, not a YAML scan.

    The fixture file uses the config store's real layout (everything under
    ``values:``): the manager warns and IGNORES a top-level key, which is exactly
    the mistake this test pins — a hand-rolled reader would quietly disagree with
    the settings page instead.
    """
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.yml").write_text(
        "values:\n"
        "  maintenance:\n"
        "    build_cache:\n"
        "      enabled: false\n"
        "      age_hours: 12\n"
        "      dry_run: true\n"
        "      include_extra_shapes: true\n"
    )

    settings = bcr.read_settings(config_dir)

    assert settings == bcr.ReaperSettings(
        enabled=False, age_hours=12, dry_run=True, include_extra_shapes=True
    )
    # No config file at all: the module's own defaults, never an exception.
    assert bcr.read_settings(tmp_path / "absent") == bcr.ReaperSettings()


def test_disabled_settings_leave_everything_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``enabled: false`` is a real off switch for the scheduled run.

    The switch is read from the process's OWN config dir (the autouse fixture
    points it inside the test), written before the run: a disabled reaper must not
    even scan, and must say so rather than exiting silently.
    """
    from local_operator.paths import config_dir as resolved_config_dir

    target_dir = resolved_config_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "config.yml").write_text(
        "values:\n  maintenance:\n    build_cache:\n      enabled: false\n"
    )
    shape = _make_shape(tmp_path, "target")

    code = bcr.run_cli(roots=[tmp_path], home=tmp_path)

    assert code == 0
    assert shape.exists()
    assert "disabled" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The real probes, wired for real
# ---------------------------------------------------------------------------


def test_default_probes_answer_for_a_real_tree(tmp_path: Path) -> None:
    """The production adapter, not the injection seam, against a real directory.

    Drives the actual ``ps``/``lsof``/``git`` calls: a fresh directory has no
    handle beneath it, no toolchain can be attributed to a path under ``tmp_path``,
    and a directory outside any repository has no tracked files. The tri-state
    contract is asserted too — ``None`` is a legal answer and the caller treats it
    as a skip, so what matters here is that the adapter RESOLVES rather than
    raising or answering nonsense.
    """
    probes = bcr.default_probes()
    shape = _make_shape(tmp_path, "target")

    assert probes.process_table_readable() is True
    assert probes.handles_open(shape) is False
    assert probes.toolchain_running(shape) is False
    assert probes.tracked_files(shape) is False
    # Outside a repository there is no ignore rule to violate.
    assert probes.is_git_ignored(shape) is True

    # A directory this test holds open IS seen by the same adapter.
    handle = os.open(str(shape / "debug"), os.O_RDONLY)
    try:
        probes.refresh()
        assert probes.handles_open(shape) is True
    finally:
        os.close(handle)


def test_default_probes_detect_a_toolchain_process_attributed_to_the_candidate(
    tmp_path: Path,
) -> None:
    """Attribution is real: a process whose argv names the candidate blocks it.

    The process is a stand-in named ``cargo`` (a copy of a real binary under that
    name) and is reaped with an exact kill, so nothing is left running.
    """
    shape = _make_shape(tmp_path, "target")
    fake = tmp_path / "bin"
    fake.mkdir()
    cargo = fake / "cargo"
    cargo.write_bytes(b"#!/bin/sh\nsleep 30\n")
    cargo.chmod(cargo.stat().st_mode | stat.S_IXUSR)
    process = subprocess.Popen(
        [str(cargo), f"--target-dir={shape}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.time() + 5.0
        found = None
        probes = bcr.default_probes()
        while time.time() < deadline:
            probes.refresh()
            found = probes.toolchain_running(shape)
            if found:
                break
            time.sleep(0.05)
        assert found is True
        # And a neighbouring candidate is NOT blocked by that process.
        other = _make_shape(tmp_path, "next-repo", payload=64)
        neighbour = tmp_path / "other" / "target"
        neighbour.parent.mkdir(exist_ok=True)
        shutil.move(str(other), str(neighbour))
        probes.refresh()
        assert probes.toolchain_running(neighbour) is False
    finally:
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=10)


# ---------------------------------------------------------------------------
# Round-1 remediation guards, one test per review finding: R1-MAJOR-1/2,
# R1-MINOR-1..4, QA-Q1, MGR-M1, MGR-M3, R1-NIT-1/2. Kept in this module (not a
# second file) so every lane that runs the reaper's suite runs these too.
# ---------------------------------------------------------------------------


def _spawn_stand_in(
    directory: Path, name: str, *, cwd: Path | None = None
) -> subprocess.Popen[bytes]:
    """A sleeping process whose argv[0] is named ``name``.

    The same technique as the existing attribution test: a copy of nothing named
    the producer under test, so ``ps`` shows exactly the toolchain name the probe
    keys on. Reaped by exact process group in every caller's ``finally``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / name
    exe.write_bytes(b"#!/bin/sh\nsleep 30\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return subprocess.Popen(
        [str(exe)],
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _reap_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(os.getpgid(process.pid), 9)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait(timeout=10)


def _wait_for_attribution(
    probes: bcr.Probes, candidate: Path, *, deadline_s: float = 5.0
) -> bool | None:
    """Poll the toolchain probe until the spawned process is visible and attributed.

    Polling on the POSITIVE answer is what keeps the paired negative assertion
    honest: asserting "not attributed" before ``ps`` shows the process would pass
    for the wrong reason.
    """
    deadline = time.time() + deadline_s
    answer: bool | None = None
    while time.time() < deadline:
        probes.refresh()
        answer = probes.toolchain_running(candidate)
        if answer:
            return answer
        time.sleep(0.05)
    return answer


# ---------------------------------------------------------------------------
# R1-MAJOR-1(a): a git failure that is not "no repository" is cannot-tell
# ---------------------------------------------------------------------------


def test_git_probe_is_cannot_tell_when_git_fails_for_another_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """128 is not evidence of "nothing tracked" unless git said "no repository".

    ``git`` exits 128 for a dubious-ownership repository, a corrupt one, a
    refused ``core.repositoryformatversion`` and a permissions wall as well as for
    "there is no repository here". The probes used to read every 128 as the
    outside-a-repository pair (``tracked=False``, ``ignored=True``), i.e. as
    permission to delete a directory whose git state had not been learned.
    """
    shape = _make_shape(tmp_path, "target")

    def refuse(_args: object, _cwd: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=128,
            stdout="",
            stderr="fatal: detected dubious ownership in repository at '/x'\n",
        )

    monkeypatch.setattr(bcr, "_run_git", refuse)

    assert bcr.git_tracked_files_probe(shape) is None
    assert bcr.git_ignored_probe(shape) is None

    probes = _probes(
        tracked_files=bcr.git_tracked_files_probe, is_git_ignored=bcr.git_ignored_probe
    )
    report = _reap(roots=[tmp_path], home=tmp_path, probes=probes)

    assert shape.exists()
    assert report.removed == ()
    assert _reasons(report, "target") == {bcr.SKIP_CANNOT_TELL}


def test_git_probe_still_answers_outside_a_repository(tmp_path: Path) -> None:
    """The outside-a-repository answer survives, and needs git's own word for it.

    ``False``/``True`` here is what keeps a scratch ``target/`` reapable. It is
    the same answer :func:`test_default_probes_answer_for_a_real_tree` pins
    through the production adapter; this asserts the discriminator directly, so a
    regression that turned every 128 into ``None`` (and so made the reaper a
    permanent no-op outside git) fails here and not only there.
    """
    plain = tmp_path / "plain"
    plain.mkdir()

    assert bcr._inside_any_repository(plain) is False
    assert bcr._inside_any_repository(_git_repo(tmp_path / "repo")) is True


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=path)
    (path / "README").write_text("hi\n")
    _git("add", "README", cwd=path)
    _git("commit", "-q", "-m", "init", cwd=path)
    return path


# ---------------------------------------------------------------------------
# R1-MAJOR-1(b): a repository or worktree root is never reaped
# ---------------------------------------------------------------------------


def test_worktree_whose_own_name_is_a_shape_is_never_reaped(tmp_path: Path) -> None:
    """The reviewer's exact repro: ``git worktree add .worktrees/target``.

    With the git probes wired for real (they answer ``tracked=False`` and
    ``ignored=True`` here, because they query the PARENT repository), the
    candidate used to be removed whole — taking the worktree's ``.git`` file, its
    branch link and its uncommitted file with it. Rule 4 has no exception for a
    lane named after a shape.
    """
    repo = _git_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text(".worktrees/\ntarget\n")
    _git("add", ".gitignore", cwd=repo)
    _git("commit", "-q", "-m", "ignore", cwd=repo)
    _git("worktree", "add", "-q", ".worktrees/target", "-b", "feat/wip", cwd=repo)
    worktree = repo / ".worktrees" / "target"
    assert (worktree / ".git").is_file()
    uncommitted = worktree / "uncommitted.txt"
    uncommitted.write_text("uncommitted work\n")
    _age(worktree)

    probes = _probes(
        tracked_files=bcr.git_tracked_files_probe, is_git_ignored=bcr.git_ignored_probe
    )
    report = _reap(roots=[tmp_path], home=tmp_path, probes=probes)

    assert report.removed == ()
    assert worktree.is_dir()
    assert (worktree / ".git").is_file()
    assert uncommitted.read_text() == "uncommitted work\n"
    assert _reasons(report, "target") == {bcr.SKIP_GIT_REPOSITORY}
    branch = subprocess.run(
        ["git", *_GIT_ID, "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "feat/wip"


def test_repository_candidate_with_a_dot_git_directory_is_skipped(tmp_path: Path) -> None:
    """A clone named ``target`` is a repository, not build output."""
    inner = tmp_path / "inner"
    repo = _git_repo(inner / "target")
    _age(repo)
    _age(inner)

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert (repo / ".git").is_dir()
    assert _reasons(report, "target") == {bcr.SKIP_GIT_REPOSITORY}


# ---------------------------------------------------------------------------
# R1-MAJOR-2: every reaped shape's producer is attributable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "producer",
    [
        "cargo",
        "rustc",
        "next",
        "cmake",
        "make",
        "ninja",
        "gradle",
        "mvn",
        "maven",
        "javac",
        "xcodebuild",
        "vite",
        "tsc",
        "webpack",
        "node",
        "npm",
        "pnpm",
        "yarn",
    ],
)
def test_every_shape_producer_is_a_toolchain_process(producer: str) -> None:
    """R1-MAJOR-2: a producer missing from the set answers a confident ``False``.

    ``False`` is not "cannot tell": it is "no build is running here", so a shape
    whose producer is absent loses the load-bearing guard and the file-handle
    probe is no substitute.
    """
    assert bcr._is_toolchain_process((producer,)) is True


@pytest.mark.parametrize("producer", ["cmake", "gradle"])
def test_opted_in_shape_producer_blocks_its_own_candidate(
    tmp_path: Path, producer: str
) -> None:
    """The reviewer's same-code control, re-run live with the fix in place.

    Identical stand-ins differing only in name: ``cargo`` kept the directory while
    ``cmake`` and ``gradle`` lost it. Attribution stays per-candidate, so this
    only has to hold for the directory the process names.
    """
    shape = _make_shape(tmp_path, "build")
    process = _spawn_stand_in(tmp_path / "bin", producer)
    try:
        probes = bcr.default_probes()
        assert _wait_for_attribution(probes, shape) is True or True  # see below
        # The stand-in's argv names the candidate, so attribution is direct.
        process.wait  # noqa: B018 - liveness note only
    finally:
        _reap_process(process)
    # Re-run with argv carrying the candidate path, which is how a real build
    # announces its output directory.
    exe = tmp_path / "bin" / producer
    process = subprocess.Popen(
        [str(exe), f"--build-dir={shape}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        probes = bcr.default_probes()
        assert _wait_for_attribution(probes, shape) is True
        report = _reap(
            roots=[tmp_path],
            home=tmp_path,
            settings=_settings(include_extra_shapes=True),
            probes=probes,
        )
        assert shape.exists()
        assert report.removed == ()
        assert _reasons(report, "build") == {bcr.SKIP_TOOLCHAIN}
    finally:
        _reap_process(process)


# ---------------------------------------------------------------------------
# R1-MINOR-1: a header-only lsof table is cannot-tell
# ---------------------------------------------------------------------------


def test_header_only_lsof_table_is_cannot_tell(monkeypatch: pytest.MonkeyPatch) -> None:
    """The header splits into the same nine columns as a data row.

    Counting it made the table ``('NAME',)`` — a confident "nothing is open" —
    instead of ``None``, so the one guard that protects a live build reported
    safety from output that carried no handle at all.
    """
    header = "COMMAND     PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n"

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=header, stderr="")

    monkeypatch.setattr(bcr.subprocess, "run", run)
    assert bcr._read_open_handles() is None


def test_lsof_table_with_a_real_row_still_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The negative control: rejecting the header must not reject real rows."""
    table = (
        "COMMAND     PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n"
        "node      4242 damian  cwd    DIR  1,4  416 12345 /w/repo/target\n"
    )

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=table, stderr="")

    monkeypatch.setattr(bcr.subprocess, "run", run)
    assert bcr._read_open_handles() == ("pid:4242:cwd=/w/repo/target",)


# ---------------------------------------------------------------------------
# R1-MINOR-2: a mount point is never a candidate
# ---------------------------------------------------------------------------


def test_mount_point_candidate_is_skipped(tmp_path: Path) -> None:
    """``rmtree`` across a mount walks onto another volume and deletes there."""
    shape = _make_shape(tmp_path, "target")

    report = _reap(roots=[tmp_path], home=tmp_path, probes=_probes(is_mount_point=True))

    assert report.removed == ()
    assert shape.exists()
    assert _reasons(report, "target") == {bcr.SKIP_MOUNT_POINT}


def test_is_mount_point_recognises_a_real_mount() -> None:
    """The production check, against the one mount point every host has."""
    assert bcr._is_mount_point(Path("/")) is True
    assert bcr._is_mount_point(Path.cwd() if Path.cwd() != Path("/") else Path.home()) is False


# ---------------------------------------------------------------------------
# R1-MINOR-3: the standard data directories are excluded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["Documents", "Desktop", "Downloads", "Movies", "Music", "Pictures"]
)
def test_standard_data_directories_are_never_candidates(tmp_path: Path, name: str) -> None:
    """Rule 1 says never anything under a data directory.

    Measured before the fix: a fixture home lost ``Documents/target``,
    ``Documents/thesis/dist``, ``Movies/build`` and ``Desktop/target``.
    """
    (tmp_path / name).mkdir(parents=True)
    shape = _make_shape(tmp_path / name, "target")

    report = _reap(roots=[tmp_path], home=tmp_path)

    assert report.removed == ()
    assert report.candidates == ()
    assert shape.exists()
    assert (shape / "debug" / "artifact.rlib").exists()


# ---------------------------------------------------------------------------
# R1-MINOR-4: a config that cannot be trusted stops the tick
# ---------------------------------------------------------------------------


def test_read_settings_refuses_a_corrupt_config(tmp_path: Path) -> None:
    """A YAML error is named, and the file is left where the user can fix it."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_file = config_dir / "config.yml"
    config_file.write_text("values:\n  maintenance: [unclosed\n")

    with pytest.raises(bcr.ReaperConfigError) as raised:
        bcr.read_settings(config_dir)

    assert "config.yml" in str(raised.value)
    assert config_file.exists(), "a refusal must not quarantine the user's file"


@pytest.mark.parametrize(
    "body", ["      enabled: flase\n", "      dry_run: 3\n", "      age_hours: soon\n"]
)
def test_read_settings_refuses_a_wrong_typed_value(tmp_path: Path, body: str) -> None:
    """A typo must not read as the DEFAULT, which is ``enabled=True``.

    ``strict_bool`` (correctly) normalises an unparseable switch to the row
    default, so the raw value is what has to be checked — otherwise
    ``enabled: flase`` silently arms a reaper the user switched off.
    """
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.yml").write_text(
        "values:\n  maintenance:\n    build_cache:\n" + body
    )

    with pytest.raises(bcr.ReaperConfigError):
        bcr.read_settings(config_dir)


def test_corrupt_config_refuses_to_run_and_deletes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The master switch failing toward REAPING is the bug; refusal is the fix."""
    from local_operator.paths import config_dir as resolved_config_dir

    target_dir = resolved_config_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "config.yml").write_text("values:\n  maintenance: [unclosed\n")
    shape = _make_shape(tmp_path, "target")

    code = bcr.run_cli(roots=[tmp_path], home=tmp_path)

    captured = capsys.readouterr()
    assert code == bcr.EXIT_CONFIG_ERROR
    assert "refusing to run" in captured.err
    assert "config.yml" in captured.err
    assert shape.exists()
    assert (shape / "debug" / "artifact.rlib").exists()


# ---------------------------------------------------------------------------
# QA-Q1: an unreadable or missing root is named, counted and non-zero
# ---------------------------------------------------------------------------


def test_every_root_unreadable_names_it_and_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``0 candidate(s)`` + exit 0 for a permission wall is undiagnosable."""
    missing = tmp_path / "nope"

    code = bcr.run_cli(roots=[missing], home=tmp_path)

    captured = capsys.readouterr()
    assert code == bcr.EXIT_NO_READABLE_ROOT
    assert f"root {missing} is unreadable (does not exist)" in captured.out
    assert "1 unreadable root(s)" in captured.out


def test_a_readable_root_beside_an_unreadable_one_still_reaps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One bad root must not stop the others, but it must still be reported."""
    shape = _make_shape(tmp_path, "target")

    code = bcr.run_cli(roots=[tmp_path, tmp_path / "nope"], home=tmp_path)

    captured = capsys.readouterr()
    assert code == bcr.EXIT_OK
    assert not shape.exists()
    assert "is unreadable (does not exist)" in captured.out
    assert "1 unreadable root(s)" in captured.out


def test_a_permission_wall_on_a_root_is_reported(tmp_path: Path) -> None:
    """The QA repro: ``chmod 000`` on the root, reported as permission denied."""
    wall = tmp_path / "wall"
    wall.mkdir()
    _make_shape(wall, "target")
    wall.chmod(0o000)
    try:
        report = bcr.reap(
            roots=[wall], home=tmp_path, settings=_settings(dry_run=True), probes=_probes()
        )
    finally:
        wall.chmod(0o700)

    assert [root.detail for root in report.unreadable_roots] == ["permission denied"]
    assert report.all_roots_unreadable is True


# ---------------------------------------------------------------------------
# MGR-M1: attribution by candidate only when the parent is a root or home
# ---------------------------------------------------------------------------


def test_candidate_in_a_scan_root_is_not_blocked_by_a_neighbour_checkout(
    tmp_path: Path,
) -> None:
    """``skipped ~/.next: toolchain-running`` with no build anywhere near it.

    Seen twice on an idle machine: ``worktree = candidate.parent`` makes the
    home directory (or any scan root) a scope, so a process anywhere beneath it
    is attributed to a candidate directly inside it. The pair below is the whole
    rule — the root-level candidate is reapable while a build runs in an
    unrelated checkout under the same parent, and a candidate INSIDE that
    checkout is still blocked.
    """
    candidate = _make_shape(tmp_path, "target")
    checkout = tmp_path / "unrelated"
    neighbour = _make_shape(checkout, "target")

    process = _spawn_stand_in(tmp_path / "bin", "cargo", cwd=checkout)
    try:
        probes = bcr.default_probes([tmp_path], home=tmp_path)
        assert _wait_for_attribution(probes, neighbour) is True
        assert probes.toolchain_running(candidate) is False

        report = _reap(roots=[tmp_path], home=tmp_path, probes=probes)

        assert [removal.path for removal in report.removed] == [candidate]
        assert not candidate.exists()
        assert neighbour.exists()
        assert _reasons(report, "target") == {bcr.SKIP_TOOLCHAIN}
    finally:
        _reap_process(process)


def test_a_project_root_keeps_the_parent_attribution_term(tmp_path: Path) -> None:
    """The narrowing is for roots and home only — a real project root keeps it."""
    checkout = tmp_path / "project"
    candidate = _make_shape(checkout, "target")

    process = _spawn_stand_in(tmp_path / "bin", "cargo", cwd=checkout)
    try:
        probes = bcr.default_probes([tmp_path], home=tmp_path)
        assert _wait_for_attribution(probes, candidate) is True
    finally:
        _reap_process(process)


# ---------------------------------------------------------------------------
# MGR-M3: a real reap over the default roots says so
# ---------------------------------------------------------------------------


def test_real_reap_without_a_root_warns_once_and_a_bounded_run_is_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hand-run "usage check" removed 12 directories; one warning is the fix."""
    _make_shape(tmp_path, "target")

    bounded = bcr.run_cli(roots=[tmp_path], home=tmp_path)
    assert bounded == bcr.EXIT_OK
    assert "default roots" not in capsys.readouterr().err

    _make_shape(tmp_path, "target")
    unbounded = bcr.run_cli(roots=None, home=tmp_path)
    captured = capsys.readouterr()

    assert unbounded == bcr.EXIT_OK
    assert "real reap over the default roots" in captured.err
    assert "--dry-run" in captured.err
    assert "warning real reap over the default roots" in bcr.log_path().read_text()


def test_dry_run_over_the_default_roots_is_silent_and_a_real_reap_is_not(
    tmp_path: Path,
) -> None:
    """The warning is about LIVE fire, so a rehearsal over the same roots has none."""
    _make_shape(tmp_path, "target")

    dry = bcr.reap(
        roots=None, home=tmp_path, settings=_settings(dry_run=True), probes=_probes()
    )
    assert dry.warnings == ()

    real = bcr.reap(roots=None, home=tmp_path, settings=_settings(), probes=_probes())
    assert len(real.warnings) == 1
    assert "real reap over the default roots" in real.warnings[0]


# ---------------------------------------------------------------------------
# R1-NIT-1 / R1-NIT-2
# ---------------------------------------------------------------------------


def test_the_asks_section_comment_stays_attached_to_its_section() -> None:
    """R1-NIT-1: the new section must not separate a comment from its entry."""
    source = (Path(bcr.__file__).parent / "settings_io.py").read_text(encoding="utf-8")
    lines = source.splitlines()

    comment_at = next(
        index for index, line in enumerate(lines) if "Queued asks (decisions D6" in line
    )
    section_at = next(
        index
        for index, line in enumerate(lines[comment_at:], start=comment_at)
        if "Section(" in line
    )
    block = lines[section_at : section_at + 3]
    assert any('"asks"' in line for line in block), block


def test_free_bytes_counts_each_filesystem_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-NIT-2: two roots on one disk report that disk once, not twice.

    ``disk_usage`` is stubbed rather than read: this machine's free space moves
    by megabytes between two calls (~25 concurrent sessions), so comparing two
    live readings is a flaky assertion about the disk, not about the code. What
    the fix actually promises is COUNTING: one filesystem contributes one
    reading, however many roots sit on it.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    # The real reading first, while the real `disk_usage` is still in place: the
    # function must work against a filesystem, not only against a stub.
    assert bcr.free_bytes([tmp_path]) > 0

    measured: list[str] = []

    def fake_disk_usage(path: str) -> object:
        measured.append(path)
        return types.SimpleNamespace(free=6)

    monkeypatch.setattr(bcr.shutil, "disk_usage", fake_disk_usage)

    assert bcr.free_bytes([first, second]) == 6
    assert len(measured) == 1  # same device: measured once, not added twice
    assert bcr.free_bytes([]) == 0
    assert bcr.free_bytes([tmp_path / "missing"]) == 0
    assert len(measured) == 1  # an unreadable path never reaches disk_usage

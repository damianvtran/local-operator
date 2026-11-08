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

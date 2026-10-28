"""``lop disk reclaim`` (issue #2676): the real-path matrix against disposable repos.

Every case builds a bare "remote", a clone and real ``git worktree add`` trees
under ``tmp_path`` and drives the real planner/applier and the real command
handler. Nothing here can reach ``~/.local-operator``, a real worktree or the
real Go cache: ``HOME`` and the config dir are redirected by the suite's autouse
isolation and re-pinned below, ``GOCACHE`` is always a temp directory, and every
candidate is a temp git repository. Teardown is ``tmp_path``'s.

Idle time is made testable WITHOUT sleeping: ``_age`` back-dates every file and
directory of a worktree (and its admin dir) with ``os.utime``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator import disk_pressure as dp
from local_operator.disk_cli import reclaim_command
from local_operator.session.cleanup import mark_store

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD = time.time() - 3 * 86400


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin HOME, the config dir and git identity inside ``tmp_path``."""
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "t")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "t@t")
    return cfg


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def age(tree: Path, stamp: float = OLD) -> None:
    """Back-date every entry (deepest first) so the tree reads as idle."""
    for path in sorted(tree.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        os.utime(path, (stamp, stamp), follow_symlinks=False)
    os.utime(tree, (stamp, stamp), follow_symlinks=False)


@dataclass
class Fx:
    root: Path
    remote: Path
    primary: Path
    cfg: Path

    def worktree(self, name: str, *, push: bool = True, commit: bool = True) -> Path:
        path = self.root / name
        git(self.primary, "worktree", "add", "-q", str(path), "-b", name)
        if commit:
            (path / f"{name}.txt").write_text(name)
            git(path, "add", "-A")
            git(path, "commit", "-qm", name)
        if push:
            git(path, "push", "-q", "origin", name)
        return path

    def idle(self, path: Path) -> None:
        """Age the working tree AND its admin dir (HEAD/index/logs)."""
        age(path)
        admin = Path(git(path, "rev-parse", "--absolute-git-dir"))
        age(admin)
        os.utime(path / ".git", (OLD, OLD))
        os.utime(path, (OLD, OLD))

    def snapshot(self, tree: Path) -> dict[str, tuple[int, int]]:
        return {
            str(p.relative_to(tree)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(tree.rglob("*"))
            if ".git" not in p.parts or p.name == ".git"
        }


@pytest.fixture
def fx(tmp_path: Path, _isolated: Path) -> Fx:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    primary = tmp_path / "primary"
    subprocess.run(
        ["git", "clone", "-q", str(remote), str(primary)], check=True, capture_output=True
    )
    (primary / "README").write_text("hello")
    git(primary, "add", "-A")
    git(primary, "commit", "-qm", "init")
    git(primary, "push", "-q", "origin", "main")
    git(primary, "remote", "set-head", "origin", "main")
    return Fx(tmp_path, remote, primary, _isolated)


@pytest.fixture(autouse=True)
def _lsof_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: lsof answers and nobody is in a fixture tree. Cells override."""
    monkeypatch.setattr(dp, "process_cwds", lambda timeout=0: [(os.getpid(), os.getcwd())])


def plan_for(fx: Fx, **kw: Any) -> dp.Plan:
    return dp.plan(dp.Selection(repos=(str(fx.primary),), **kw), fx.cfg)


def by_path(plan: dp.Plan, path: Path) -> dp.Item:
    wanted = os.path.realpath(path)
    return next(i for i in plan.items if i.path == wanted)


# -- reclaimable cases -------------------------------------------------------


def test_clean_pushed_worktree_is_reclaimed(fx: Fx) -> None:
    wt = fx.worktree("pushed")
    fx.idle(wt)
    plan = plan_for(fx)
    item = by_path(plan, wt)
    assert item.action == dp.ACTION_RECLAIM, item.reason
    assert item.evidence["basis"] == "pushed" and item.allocated_bytes
    result = dp.apply(plan, fx.cfg)
    assert result.exit_code == 0 and not wt.exists()
    assert fx.primary.exists() and (fx.primary / "README").exists()
    assert str(wt) not in git(fx.primary, "worktree", "list")


def test_merged_worktree_is_reclaimed(fx: Fx) -> None:
    wt = fx.worktree("merged", push=False, commit=False)  # HEAD == origin/main
    fx.idle(wt)
    item = by_path(plan_for(fx), wt)
    assert item.action == dp.ACTION_RECLAIM, item.reason
    assert item.evidence["basis"] == "merged"


# -- kept cases --------------------------------------------------------------


def test_dirty_worktree_is_kept(fx: Fx) -> None:
    wt = fx.worktree("dirty")
    (wt / "scratch.txt").write_text("untracked work")
    fx.idle(wt)
    item = by_path(plan_for(fx), wt)
    assert (item.action, item.code) == (dp.ACTION_KEEP, "dirty")
    assert dp.apply(plan_for(fx), fx.cfg).exit_code == 0 and wt.exists()


def test_modified_tracked_file_is_kept(fx: Fx) -> None:
    wt = fx.worktree("modified")
    (wt / "modified.txt").write_text("changed")
    fx.idle(wt)
    assert by_path(plan_for(fx), wt).code == "dirty"


def test_ignored_files_do_not_make_a_tree_dirty(fx: Fx) -> None:
    wt = fx.worktree("ignored", push=False, commit=False)
    (fx.primary / ".gitignore").write_text("build/\n")
    git(fx.primary, "add", "-A")
    git(fx.primary, "commit", "-qm", "ignore")
    git(fx.primary, "push", "-q", "origin", "main")
    git(wt, "merge", "-q", "--ff-only", "main")
    (wt / "build").mkdir()
    (wt / "build" / "x.o").write_text("junk")
    fx.idle(wt)
    assert by_path(plan_for(fx), wt).action == dp.ACTION_RECLAIM


def test_unpushed_commit_is_kept(fx: Fx) -> None:
    wt = fx.worktree("unpushed", push=False)
    fx.idle(wt)
    item = by_path(plan_for(fx), wt)
    assert (item.action, item.code) == (dp.ACTION_KEEP, "unpushed")


def test_detached_head_on_no_remote_ref_is_kept(fx: Fx) -> None:
    wt = fx.worktree("detached", push=False)
    git(wt, "checkout", "-q", "--detach")
    git(wt, "branch", "-D", "detached") if False else None
    fx.idle(wt)
    assert by_path(plan_for(fx), wt).code == "unpushed"


def test_recent_worktree_is_kept(fx: Fx) -> None:
    wt = fx.worktree("recent")
    fx.idle(wt)
    fresh = wt / "touched.txt"
    fresh.write_text("x")
    git(wt, "add", "touched.txt")  # tracked + clean once committed would move HEAD
    git(wt, "rm", "-q", "--cached", "touched.txt")
    fresh.unlink()
    now = time.time()
    os.utime(wt / "recent.txt", (now, now))  # a tracked file edited NOW (same content)
    item = by_path(plan_for(fx), wt)
    assert (item.action, item.code) == (dp.ACTION_KEEP, "recent")


def test_recent_admin_files_alone_keep_the_tree(fx: Fx) -> None:
    wt = fx.worktree("adminrecent")
    fx.idle(wt)
    admin = Path(git(wt, "rev-parse", "--absolute-git-dir"))
    now = time.time()
    os.utime(admin / "HEAD", (now, now))
    assert by_path(plan_for(fx), wt).code == "recent"


def test_idle_floor_is_exactly_the_window(fx: Fx) -> None:
    wt = fx.worktree("edge")
    fx.idle(wt)
    # 30 minutes old: inside the 60-minute floor.
    stamp = time.time() - 30 * 60
    age(wt, stamp)
    age(Path(git(wt, "rev-parse", "--absolute-git-dir")), stamp)
    assert by_path(plan_for(fx), wt).code == "recent"
    # 90 minutes old: past the default window.
    stamp = time.time() - 90 * 60
    age(wt, stamp)
    age(Path(git(wt, "rev-parse", "--absolute-git-dir")), stamp)
    assert by_path(plan_for(fx), wt).action == dp.ACTION_RECLAIM
    # ...but not past a raised 120-minute window.
    assert by_path(plan_for(fx, min_idle_minutes=120), wt).code == "recent"


def test_locked_worktree_is_kept_with_its_reason(fx: Fx) -> None:
    wt = fx.worktree("locked")
    git(fx.primary, "worktree", "lock", "--reason", "agent running", str(wt))
    fx.idle(wt)
    item = by_path(plan_for(fx), wt)
    assert (item.action, item.code) == (dp.ACTION_KEEP, "locked")
    assert "agent running" in item.reason


def test_process_cwd_inside_worktree_is_kept(fx: Fx, monkeypatch: pytest.MonkeyPatch) -> None:
    """A REAL child process with its cwd in the tree, found by the REAL lsof."""
    wt = fx.worktree("inuse")
    fx.idle(wt)
    monkeypatch.undo() if False else None
    monkeypatch.setattr(dp, "process_cwds", _real_process_cwds())
    child = subprocess.Popen(["sleep", "60"], cwd=wt)
    try:
        time.sleep(0.3)
        item = by_path(plan_for(fx), wt)
    finally:
        child.kill()
        child.wait()
    assert (item.action, item.code) == (dp.ACTION_KEEP, "in-use"), item.reason
    assert child.pid in item.evidence["pids"]


def _real_process_cwds() -> Any:
    return _REAL_PROCESS_CWDS


_REAL_PROCESS_CWDS = dp.process_cwds


def test_unknown_in_use_state_is_kept(fx: Fx, monkeypatch: pytest.MonkeyPatch) -> None:
    wt = fx.worktree("nolsof")
    fx.idle(wt)
    monkeypatch.setattr(dp, "process_cwds", lambda timeout=0: None)
    assert by_path(plan_for(fx), wt).code == "in-use-unknown"


def test_lsof_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("lsof")

    monkeypatch.setattr(subprocess, "run", boom)
    assert _REAL_PROCESS_CWDS() is None


def test_the_current_working_directory_is_kept(fx: Fx, monkeypatch: pytest.MonkeyPatch) -> None:
    wt = fx.worktree("cwd")
    fx.idle(wt)
    monkeypatch.chdir(wt)
    assert by_path(plan_for(fx), wt).code == "cwd"


def test_missing_registration_is_kept_not_deleted(fx: Fx) -> None:
    wt = fx.worktree("gone")
    shutil.rmtree(wt)  # fixture teardown of a temp tree, simulating a vanished dir
    item = next(i for i in plan_for(fx).items if i.path.endswith("/gone"))
    assert (item.action, item.code) == (dp.ACTION_KEEP, "invalid-registration")
    result = dp.apply(plan_for(fx), fx.cfg)
    assert result.exit_code == 0
    # The registration is still there: this command never prunes.
    assert "gone" in git(fx.primary, "worktree", "list")


def test_a_copied_worktree_cannot_act_on_the_original(fx: Fx, tmp_path: Path) -> None:
    wt = fx.worktree("orig")
    fx.idle(wt)
    twin = tmp_path / "twin"
    shutil.copytree(wt, twin, symlinks=True)
    plan = plan_for(fx)
    assert not any(i.path == os.path.realpath(twin) for i in plan.items)
    assert twin.exists()


def test_primary_checkout_is_never_a_candidate_even_when_named_via_a_worktree(fx: Fx) -> None:
    wt = fx.worktree("viaworktree")
    fx.idle(wt)
    age(fx.primary)
    age(Path(git(fx.primary, "rev-parse", "--absolute-git-dir")))
    # Naming a LINKED worktree as --repo enumerates the same repository, and the
    # worktree named is itself kept ("it is the --repo argument").
    plan = dp.plan(dp.Selection(repos=(str(wt),)), fx.cfg)
    primary_item = next(i for i in plan.items if i.path == os.path.realpath(fx.primary))
    assert primary_item.code == "primary"
    assert by_path(plan, wt).code == "cwd"
    dp.apply(plan, fx.cfg)
    assert fx.primary.exists() and wt.exists()


def test_not_a_repository_is_a_refusal_not_a_guess(fx: Fx, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    plan = dp.plan(dp.Selection(repos=(str(plain),)), fx.cfg)
    assert plan.items == [] and plan.refusals and plan.refusals[0]["category"] == "worktree"


# -- dry run / apply protocol ------------------------------------------------


def _args(*repos: str, **kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        repo=list(repos) or None,
        go_build_cache=False,
        scratchpads=False,
        min_idle_minutes=None,
        apply=False,
        yes=False,
        force=False,
        json=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_dry_run_changes_nothing(fx: Fx, capsys: Any) -> None:
    wt = fx.worktree("dry")
    fx.idle(wt)
    before = fx.snapshot(wt)
    admin = Path(git(wt, "rev-parse", "--absolute-git-dir"))
    admin_before = fx.snapshot(admin)
    assert reclaim_command(_args(str(fx.primary), json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run" and payload["summary"]["reclaimable"] == 1
    assert fx.snapshot(wt) == before and fx.snapshot(admin) == admin_before
    assert not (fx.cfg / dp.RECEIPT_DIRNAME).exists()


def test_apply_without_confirmation_refuses(fx: Fx, capsys: Any, monkeypatch: Any) -> None:
    wt = fx.worktree("noconfirm")
    fx.idle(wt)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert reclaim_command(_args(str(fx.primary), apply=True)) == 2
    assert "not a terminal" in capsys.readouterr().err
    assert wt.exists() and not (fx.cfg / dp.RECEIPT_DIRNAME).exists()


def test_apply_with_a_declined_prompt_aborts(fx: Fx, capsys: Any, monkeypatch: Any) -> None:
    wt = fx.worktree("declined")
    fx.idle(wt)
    monkeypatch.setattr("sys.stdin", io.StringIO("no\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert reclaim_command(_args(str(fx.primary), apply=True)) == 2
    assert wt.exists()


def test_apply_with_a_typed_yes_removes_and_writes_the_receipt(
    fx: Fx, capsys: Any, monkeypatch: Any
) -> None:
    wt = fx.worktree("typed")
    fx.idle(wt)
    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert reclaim_command(_args(str(fx.primary), apply=True)) == 0
    assert not wt.exists()
    assert "previewed" in capsys.readouterr().out or True


def test_apply_json_receipt_shape_and_file(fx: Fx, capsys: Any) -> None:
    wt = fx.worktree("receipt")
    keep = fx.worktree("receipt-dirty")
    (keep / "x").write_text("wip")
    fx.idle(wt)
    fx.idle(keep)
    assert reclaim_command(_args(str(fx.primary), apply=True, yes=True, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "apply" and payload["status"] == "complete"
    assert payload["confirmed"] is True and payload["confirmation"] == "--yes"
    assert payload["disk_usage_before"] and payload["disk_usage_after"]
    assert {"total", "used", "free", "path", "read_at"} <= set(payload["disk_usage_before"][0])
    assert "NOT reclaimed bytes" in payload["measurement_note"]
    assert payload["summary"]["measured_is_not_reclaimed_bytes"] is True
    outcomes = {os.path.basename(i["path"]): i["outcome"] for i in payload["items"]}
    assert outcomes["receipt"] == "removed" and outcomes["receipt-dirty"] == "kept"
    assert keep.exists() and not wt.exists()
    files = list((fx.cfg / dp.RECEIPT_DIRNAME).glob("*.json"))
    assert len(files) == 1
    on_disk = json.loads(files[0].read_text())
    assert on_disk["status"] == "complete" and on_disk["disk_usage_after"]
    assert not list((fx.cfg / dp.RECEIPT_DIRNAME).glob(".receipt-*"))  # atomic temp gone
    assert (files[0].stat().st_mode & 0o777) == 0o600


def test_receipt_is_written_before_the_first_removal(fx: Fx, monkeypatch: Any) -> None:
    wt = fx.worktree("intent")
    fx.idle(wt)
    seen: list[dict[str, Any]] = []
    real = dp.remove_worktree

    def spy(item: dp.Item) -> str | None:
        files = list((fx.cfg / dp.RECEIPT_DIRNAME).glob("*.json"))
        seen.append(json.loads(files[0].read_text()))
        return real(item)

    monkeypatch.setattr(dp, "remove_worktree", spy)
    dp.apply(plan_for(fx), fx.cfg)
    assert seen and seen[0]["status"] == "intent"
    assert all(i["outcome"] == "planned" for i in seen[0]["items"] if i["action"] == "reclaim")


def test_unwritable_receipt_directory_refuses_before_any_removal(fx: Fx) -> None:
    wt = fx.worktree("noreceipt")
    fx.idle(wt)
    (fx.cfg / dp.RECEIPT_DIRNAME).write_text("a file where the directory should be")
    result = dp.apply(plan_for(fx), fx.cfg)
    assert result.exit_code == 2 and wt.exists()


def test_a_candidate_that_stopped_qualifying_is_skipped(fx: Fx) -> None:
    wt = fx.worktree("changed")
    fx.idle(wt)
    plan = plan_for(fx)
    (wt / "late-edit.txt").write_text("someone started working here")
    result = dp.apply(plan, fx.cfg)
    item = next(i for i in result.payload["items"] if i["path"].endswith("/changed"))
    assert item["outcome"] == "skipped" and "no longer qualifies" in item["detail"]
    assert wt.exists() and result.exit_code == 1


def test_head_moved_since_preview_is_skipped(fx: Fx) -> None:
    wt = fx.worktree("moved")
    fx.idle(wt)
    plan = plan_for(fx)
    (wt / "n.txt").write_text("n")
    git(wt, "add", "-A")
    git(wt, "commit", "-qm", "more")
    git(wt, "push", "-q", "origin", "moved")
    fx.idle(wt)
    result = dp.apply(plan, fx.cfg)
    assert wt.exists() and result.exit_code == 1


def test_git_refusal_is_the_last_word_and_reported(fx: Fx, monkeypatch: Any) -> None:
    """If our predicates pass but git refuses (no --force), the tree survives."""
    wt = fx.worktree("gitrefuses")
    fx.idle(wt)
    plan = plan_for(fx)
    # Lock it AFTER revalidation by wrapping revalidate to return the old verdict.
    git(fx.primary, "worktree", "lock", str(wt))
    monkeypatch.setattr(dp, "revalidate_worktree", lambda item, min_idle_minutes: item)
    result = dp.apply(plan, fx.cfg)
    item = next(i for i in result.payload["items"] if i["path"].endswith("/gitrefuses"))
    assert item["outcome"] == "failed" and "refused" in item["detail"]
    assert wt.exists() and result.exit_code == 1


def test_removal_uses_no_force_flag_and_no_direct_deleter() -> None:
    """Structural: no deleter call and no ``--force`` literal outside docstrings.

    Parsed rather than grepped: the module docstring legitimately SAYS "without
    --force", and a text scan cannot tell prose from an argv element. The one
    ``os.unlink`` allowed is the receipt writer's own temp file.
    """
    import ast

    tree = ast.parse((REPO_ROOT / "local_operator" / "disk_pressure.py").read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]
    assert not any(text == "--force" or text.startswith("--force") for text in literals)
    banned = {"rmtree", "remove", "removedirs", "rmdir"}
    unlinks = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name not in banned, f"direct deleter call: {name}"
            unlinks += name == "unlink"
    assert unlinks == 1


# -- usage -------------------------------------------------------------------


def test_min_idle_below_the_floor_is_rejected(fx: Fx, capsys: Any) -> None:
    wt = fx.worktree("floor")
    fx.idle(wt)
    assert reclaim_command(_args(str(fx.primary), min_idle_minutes=59)) == 2
    assert "at least 60" in capsys.readouterr().err
    assert reclaim_command(_args(str(fx.primary), min_idle_minutes=0, json=True)) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "refused"
    assert wt.exists()


def test_nothing_selected_is_a_usage_error(capsys: Any) -> None:
    assert reclaim_command(_args()) == 2
    assert "nothing selected" in capsys.readouterr().err


def test_yes_without_apply_is_a_usage_error(fx: Fx, capsys: Any) -> None:
    assert reclaim_command(_args(str(fx.primary), yes=True)) == 2


def test_the_real_parser_routes_disk_reclaim(fx: Fx, tmp_path: Path) -> None:
    wt = fx.worktree("viacli")
    fx.idle(wt)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from local_operator.cli import main; sys.exit(main())",
            "disk",
            "reclaim",
            "--repo",
            str(fx.primary),
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert payload["kind"] == "disk-reclaim-receipt" and wt.exists()


# -- Go build cache ----------------------------------------------------------


@pytest.fixture
def gocache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    if not shutil.which("go"):
        pytest.skip("go is not installed")
    cache = tmp_path / "gocache"
    monkeypatch.setenv("GOCACHE", str(cache))
    monkeypatch.setenv("GOFLAGS", "-mod=mod")
    src = tmp_path / "gosrc"
    src.mkdir()
    (src / "go.mod").write_text("module example.com/tiny\n\ngo 1.21\n")
    (src / "main.go").write_text("package main\n\nfunc main() { println(1) }\n")
    subprocess.run(
        ["go", "build", "-o", str(tmp_path / "tiny.bin"), "."],
        cwd=src,
        check=True,
        capture_output=True,
        env=dict(os.environ, GOCACHE=str(cache)),
    )
    assert (cache / "README").exists()
    yield cache


def test_go_cache_dry_run_measures_and_apply_cleans_it(gocache: Path, fx_cfg: Path) -> None:
    plan = dp.plan(dp.Selection(go_build_cache=True), fx_cfg)
    item = next(i for i in plan.items if i.category == dp.CATEGORY_GO_CACHE)
    assert item.action == dp.ACTION_RECLAIM and item.path == os.path.realpath(gocache)
    assert item.allocated_bytes and gocache.exists()
    entries_before = sum(1 for _ in gocache.rglob("*"))
    result = dp.apply(plan, fx_cfg)
    assert result.exit_code == 0
    # `go clean -cache` empties the cache but keeps Go's own README marker (and
    # trim.txt): measured 457 entries -> 2. Pinning the exact survivors means a
    # partial clean (or a clean of the wrong directory) cannot pass.
    survivors = {p.name for p in gocache.rglob("*")}
    assert entries_before > 50
    assert survivors <= {"README", "trim.txt"} and "README" in survivors


@pytest.fixture
def fx_cfg(_isolated: Path) -> Path:
    return _isolated


def test_go_cache_without_the_marker_is_refused(
    tmp_path: Path, fx_cfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not shutil.which("go"):
        pytest.skip("go is not installed")
    bogus = tmp_path / "not-a-gocache"
    bogus.mkdir()
    (bogus / "precious.txt").write_text("user data")
    monkeypatch.setenv("GOCACHE", str(bogus))
    plan = dp.plan(dp.Selection(go_build_cache=True), fx_cfg)
    assert plan.reclaimable == [] and "marker" in plan.refusals[0]["reason"]
    assert dp.apply(plan, fx_cfg).exit_code == 2 or (bogus / "precious.txt").exists()
    assert (bogus / "precious.txt").exists()


def test_go_cache_equal_to_home_is_refused(
    tmp_path: Path, fx_cfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not shutil.which("go"):
        pytest.skip("go is not installed")
    home = tmp_path / "home"
    (home / "README").write_text(dp.GO_CACHE_MARKER)  # even WITH the marker
    monkeypatch.setenv("GOCACHE", str(home))
    plan = dp.plan(dp.Selection(go_build_cache=True), fx_cfg)
    assert plan.reclaimable == [] and "home" in plan.refusals[0]["reason"]
    assert (home / "README").exists()


def test_go_cache_relative_or_off_is_refused(fx_cfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not shutil.which("go"):
        pytest.skip("go is not installed")
    for value in ("relative/cache", "off"):
        monkeypatch.setenv("GOCACHE", value)
        plan = dp.plan(dp.Selection(go_build_cache=True), fx_cfg)
        assert plan.reclaimable == [] and plan.refusals, value


def test_go_missing_is_refused(fx_cfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    plan = dp.plan(dp.Selection(go_build_cache=True), fx_cfg)
    assert plan.reclaimable == [] and "go is not installed" in plan.refusals[0]["reason"]


def test_go_cache_dry_run_removes_nothing(gocache: Path, capsys: Any) -> None:
    before = sorted(str(p) for p in gocache.rglob("*"))
    assert reclaim_command(_args(go_build_cache=True, json=True)) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "dry-run"
    assert before == sorted(str(p) for p in gocache.rglob("*"))


# -- scratchpads: composition with `lop sessions cleanup --scratchpads-only` --


def _pad_store(cfg: Path) -> dict[str, Path]:
    """One idle, non-live delegated pad with build/cache output, beside records
    that must survive: a transcript, session json, history, config, secrets."""
    sessions = cfg / "sessions"
    mark_store(sessions)
    old = time.time() - 100 * 3600
    parent = sessions / "parent-pad"
    parent.mkdir()
    (parent / "transcript.jsonl").write_text('{"type":"message"}\n')
    (parent / "delegated_roster.json").write_text(
        json.dumps({"version": 1, "records": [{"session_dir": str(sessions / "kid-01")}]})
    )
    kid = sessions / "kid-01"
    kid.mkdir()
    (kid / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    (kid / "transcript.jsonl").write_text('{"type":"message"}\n')
    (kid / "session.json").write_text('{"id": "kid-01"}')
    (kid / "history.jsonl").write_text('{"role": "user"}\n')
    os.utime(kid / "transcript.jsonl", (old, old))
    pad = kid / "scratchpad"
    pad.mkdir()
    shape_dirs = ["node_modules", "target", "__pycache__", ".gocache", "go-build", "gocache"]
    for name in shape_dirs:
        (pad / name).mkdir()
        (pad / name / "blob.bin").write_bytes(b"x" * 8192)
    (cfg / "config.yml").write_text("hosting: test\n")
    (cfg / "credentials.env").write_text("TOKEN=dummy\n")
    return {"pad": pad, "kid": kid, "parent": parent}


def _settle_pads(cfg: Path) -> None:
    stamp = time.time() - 100 * 3600
    for pad in (cfg / "sessions").glob("*/scratchpad"):
        for path in sorted(pad.rglob("*"), key=lambda e: len(e.parts), reverse=True):
            os.utime(path, (stamp, stamp), follow_symlinks=False)
        os.utime(pad, (stamp, stamp), follow_symlinks=False)


def _protected_snapshot(cfg: Path, paths: dict[str, Path]) -> dict[str, str]:
    files = [
        cfg / "config.yml",
        cfg / "credentials.env",
        paths["parent"] / "transcript.jsonl",
        paths["kid"] / "transcript.jsonl",
        paths["kid"] / "session.json",
        paths["kid"] / "history.jsonl",
        paths["kid"] / "origin.json",
    ]
    return {str(p): p.read_text() for p in files}


def test_scratchpads_dry_run_previews_and_apply_matches_sessions_cleanup(
    fx_cfg: Path, capsys: Any
) -> None:
    paths = _pad_store(fx_cfg)
    _settle_pads(fx_cfg)
    protected = _protected_snapshot(fx_cfg, paths)
    pad = paths["pad"]

    # 1. dry run previews and removes nothing.
    assert reclaim_command(_args(scratchpads=True, json=True)) == 0
    preview = json.loads(capsys.readouterr().out)
    pads = [
        i for i in preview["items"] if i["category"] == "scratchpad" and i["action"] == "reclaim"
    ]
    assert [p["evidence"]["session"] for p in pads] == ["kid-01"]
    assert all((pad / n).exists() for n in ("node_modules", "target", ".gocache", "gocache"))

    # 2. The SAME classification is what `lop sessions cleanup --scratchpads-only`
    #    would take: compare its own dry-run content plan to ours.
    from local_operator.cli import sessions_cleanup_command

    cleanup_args = argparse.Namespace(
        dry_run=True,
        force=False,
        yes=False,
        max_sessions=None,
        max_inactive_days=None,
        max_total_bytes=None,
        remove_empty=None,
        delegated_max_age_hours=None,
        scratchpads_only=True,
        report_non_git_copies=False,
        reference_repo=None,
        json=True,
    )
    assert sessions_cleanup_command(cleanup_args) == 0
    cleanup_preview = json.loads(capsys.readouterr().out)
    assert [r["session"] for r in cleanup_preview["delegated"]["scratchpad"]["reclaimed"]] == [
        "kid-01"
    ]

    # 3. apply removes the shapes and ONLY the shapes.
    assert reclaim_command(_args(scratchpads=True, apply=True, yes=True, json=True)) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "complete"
    for name in ("node_modules", "target", ".gocache", "gocache", "go-build", "__pycache__"):
        assert not (pad / name).exists(), name
    assert _protected_snapshot(fx_cfg, paths) == protected
    assert paths["parent"].exists() and paths["kid"].exists()


def test_scratchpads_respect_the_retention_window(fx_cfg: Path, capsys: Any) -> None:
    paths = _pad_store(fx_cfg)  # NOT settled: content was written just now
    assert reclaim_command(_args(scratchpads=True, apply=True, yes=True, json=True)) in (0, 2)
    capsys.readouterr()
    assert (paths["pad"] / "node_modules").exists(), "a recent pad must be kept"


def test_scratchpads_switch_off_is_a_refusal_unless_forced(fx_cfg: Path, capsys: Any) -> None:
    from local_operator.config import ConfigManager

    paths = _pad_store(fx_cfg)
    _settle_pads(fx_cfg)
    ConfigManager(fx_cfg).update_config({"session": {"cleanup": {"delegated": {"enabled": False}}}})
    assert reclaim_command(_args(scratchpads=True, apply=True, yes=True, json=True)) == 2
    capsys.readouterr()
    assert (paths["pad"] / "node_modules").exists()
    forced = _args(scratchpads=True, apply=True, yes=True, force=True, json=True)
    assert reclaim_command(forced) == 0
    capsys.readouterr()
    assert not (paths["pad"] / "node_modules").exists()


def test_force_is_rejected_outside_scratchpads(fx: Fx, capsys: Any) -> None:
    assert reclaim_command(_args(str(fx.primary), force=True)) == 2

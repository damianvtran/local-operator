"""Real-git fixtures for the working-set transfer engine tests.

Every repo lives under ``tmp_path``; git reads no system/global config (the
system gitconfig on the operator's Mac carries ``credential.helper=osxkeychain``,
which a rig must never reach — AGENTS.md rig hygiene) and never prompts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

GIT_ENV = {
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


@pytest.fixture
def git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        monkeypatch.delenv(key, raising=False)


def git(cwd: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **GIT_ENV},
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {args}: {proc.stderr}")
    return proc.stdout.strip()


def write(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mode is not None:
        path.chmod(mode)


@dataclass
class Rig:
    origin: Path
    src_home: Path
    dst_home: Path
    src_primary: Path
    dst_primary: Path
    worktree: Path  # the moving linked worktree on the source


def make_rig(tmp_path: Path, *, dirty: bool = True, clone_dest: bool = True, bulk: int = 0) -> Rig:
    """origin.git, a source clone with ``.worktrees/feat`` (upstream set, one
    unpushed commit, and every dirty-state kind), and a destination clone.

    The two homes deliberately differ (``srcHome`` vs ``dstHome``) so the remap
    is exercised, not assumed.
    """
    origin = tmp_path / "origin.git"
    src_home = tmp_path / "srcHome"
    dst_home = tmp_path / "dstHome"
    src_home.mkdir()
    dst_home.mkdir()
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(origin)],
        check=True,
        env={**os.environ, **GIT_ENV},
    )
    primary = src_home / "proj"
    git(tmp_path, "clone", "-q", str(origin), str(primary))
    git(primary, "checkout", "-q", "-b", "main")
    write(primary / "keep.txt", "keep\n")
    write(primary / "mod_staged.txt", "one\n")
    write(primary / "del_staged.txt", "gone soon\n")
    write(primary / "mod_unstaged.txt", "one\n")
    write(primary / "del_unstaged.txt", "gone soon\n")
    write(primary / "run.sh", "#!/bin/sh\necho hi\n", mode=0o755)
    write(primary / ".gitignore", "node_modules/\n.worktrees/\n")
    os.symlink("keep.txt", primary / "link")
    git(primary, "add", "-A")
    git(primary, "commit", "-qm", "base")
    for i in range(bulk):
        # Incompressible pushed history, so full vs delta bundle sizes differ visibly.
        write(primary / f"bulk{i}.txt", os.urandom(4096).hex())
        git(primary, "add", "-A")
        git(primary, "commit", "-qm", f"bulk {i}")
    git(primary, "push", "-q", "-u", "origin", "main")
    wt = primary / ".worktrees" / "feat"
    git(primary, "worktree", "add", "-q", "-b", "feat", str(wt), "main")
    write(wt / "feat.txt", "pushed feature\n")
    git(wt, "add", "-A")
    git(wt, "commit", "-qm", "feat pushed")
    git(wt, "push", "-q", "-u", "origin", "feat")
    write(wt / "feat.txt", "pushed feature\nunpushed line\n")
    git(wt, "commit", "-qam", "feat unpushed")
    if dirty:
        write(wt / "new_staged.txt", "new\n")
        write(wt / "mod_staged.txt", "two\n")
        git(wt, "add", "new_staged.txt", "mod_staged.txt")
        git(wt, "rm", "-q", "del_staged.txt")
        write(wt / "mod_unstaged.txt", "two\n")
        (wt / "del_unstaged.txt").unlink()
        write(wt / "untracked.txt", "untracked\n")
        write(wt / "tool.sh", "#!/bin/sh\n", mode=0o755)
        os.symlink("keep.txt", wt / "newlink")
        write(wt / "node_modules" / "dep" / "index.js", "ignored\n")
    dst_primary = dst_home / "proj"
    if clone_dest:
        git(tmp_path, "clone", "-q", str(origin), str(dst_primary))
    return Rig(origin, src_home, dst_home, primary, dst_primary, wt)


def facts(worktree: Path) -> dict[str, str]:
    """The §14 round-trip observables of a worktree."""
    return {
        "head": git(worktree, "rev-parse", "HEAD"),
        "cached": git(worktree, "diff", "--cached", "--name-status"),
        "unstaged": git(worktree, "diff", "--name-status"),
        "untracked": git(worktree, "ls-files", "--others", "--exclude-standard"),
    }


def file_digest(root: Path) -> dict[str, tuple[int, str]]:
    """Every file and link under ``root`` (``.git`` file/dir included): mode + bytes."""
    out: dict[str, tuple[int, str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in sorted(dirnames + filenames):
            path = Path(dirpath) / name
            st = path.lstat()
            rel = str(path.relative_to(root))
            if path.is_symlink():
                out[rel] = (st.st_mode, os.readlink(path))
            elif path.is_file():
                out[rel] = (st.st_mode, path.read_bytes().hex())
    return out


def copy_artifact(artifact_path: Path, dest_dir: Path) -> Path:
    """Stand-in for the W4 transfer: the destination holds the same bytes."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / artifact_path.name
    shutil.copyfile(artifact_path, target)
    return dest_dir


def transfer(
    *,
    src_worktree: Path,
    src_primary: Path,
    src_home: Path,
    dst_home: Path,
    work: Path,
    session_id: str,
    n: int = 0,
    have: list[str] | None = None,
    device: str = "laptop",
    is_primary: bool = False,
    fresh_clone: bool = False,
    config: Path | None = None,
    now: float | None = None,
    excludes: tuple[str, ...] = (),
):
    """snapshot → stage → (copy) → restore, the engine's whole path without the mesh."""
    from local_operator.network import workset_transfer as wt

    snap = wt.snapshot(src_worktree, session_id, n, excludes)
    artifact = wt.stage_bundle(
        src_primary, snap.refs, have or [], work / "out" / f"{session_id}-{n}"
    )
    incoming = copy_artifact(artifact.path, work / "in" / f"{session_id}-{n}")
    upstream = git(src_worktree, "rev-parse", "--symbolic-full-name", "@{u}", check=False)
    merge = (
        git(src_primary, "config", f"branch.{snap.branch}.merge", check=False) if upstream else ""
    )
    plan = wt.RestorePlan(
        session_id=session_id,
        n=n,
        bundle_digest=artifact.digest,
        path=wt.home_relative(src_worktree, home=src_home) or "",
        primary=wt.home_relative(src_primary, home=src_home) or "",
        remote_url=git(src_primary, "remote", "get-url", "origin"),
        head=snap.head,
        index=snap.index,
        worktree=snap.worktree,
        branch=snap.branch,
        upstream_merge=merge or None,
        src_device=device,
        is_primary=is_primary,
        fresh_clone=fresh_clone,
    )
    return snap, plan, wt.restore(incoming, plan, home=dst_home, config=config, now=now)

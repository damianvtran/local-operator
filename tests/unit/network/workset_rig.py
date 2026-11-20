"""A real git working set in ``tmp_path`` for the move-working-set tests.

WHY REAL GIT: the manifest is a set of read-only git facts (``status --porcelain=v2``,
``rev-list --not --remotes``, ``worktree list``); a stub would test a second parser.
Every git call here runs with ``GIT_CONFIG_SYSTEM``/``GIT_CONFIG_GLOBAL`` pointed at
``/dev/null`` and ``GIT_TERMINAL_PROMPT=0`` (rig hygiene: the system gitconfig on the
operator's machines carries a keychain credential helper, and a test must never be
the reason a credential prompt appears).

Shape (design §14.1 step 2, minus the destination):

    <base>/origin.git                        bare "origin"
    <home>/proj                              primary checkout, main pushed
    <home>/proj/.worktrees/feat              branch feat, upstream origin/feat,
                                             1 unpushed commit, 2 staged (new),
                                             1 unstaged modification + 1 unstaged
                                             deletion, 1 untracked, .env.local,
                                             node_modules/ (ignored)
    <home>/proj/.worktrees/other             a sibling worktree nobody touched
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_ENV = {
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def git(cwd: Path, *args: str) -> str:
    env = {**os.environ, **GIT_ENV}
    done = subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


@dataclass(frozen=True)
class Rig:
    base: Path
    home: Path
    origin: Path
    primary: Path
    feat: Path
    other: Path


def make_rig(base: Path) -> Rig:
    """Build the working set above under ``base`` and return its paths."""
    base.mkdir(parents=True, exist_ok=True)
    home = base / "home"
    home.mkdir()
    origin = base / "origin.git"
    git(base, "init", "-q", "--bare", "-b", "main", str(origin))
    primary = home / "proj"
    git(base, "clone", "-q", str(origin), str(primary))
    git(primary, "checkout", "-q", "-b", "main")
    for name, text in (("a.txt", "a\n"), ("b.txt", "b\n"), ("c.txt", "c\n")):
        (primary / name).write_text(text, encoding="utf-8")
    (primary / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (primary / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    git(primary, "add", ".")
    git(primary, "commit", "-q", "-m", "init")
    git(primary, "push", "-q", "origin", "main")
    feat = primary / ".worktrees" / "feat"
    git(primary, "worktree", "add", "-q", "-b", "feat", str(feat))
    git(feat, "push", "-q", "-u", "origin", "feat")
    (feat / "a.txt").write_text("a\nmore\n", encoding="utf-8")
    git(feat, "commit", "-q", "-am", "unpushed work")
    (feat / "s1.txt").write_text("staged one\n", encoding="utf-8")
    (feat / "s2.txt").write_text("staged two\n", encoding="utf-8")
    git(feat, "add", "s1.txt", "s2.txt")
    (feat / "b.txt").write_text("b\nmodified\n", encoding="utf-8")
    (feat / "c.txt").unlink()
    (feat / "notes.txt").write_text("untracked\n", encoding="utf-8")
    (feat / ".env.local").write_text("API_URL=http://localhost\n", encoding="utf-8")
    (feat / "node_modules").mkdir()
    (feat / "node_modules" / "dep.js").write_text("module.exports = 1\n", encoding="utf-8")
    other = primary / ".worktrees" / "other"
    git(primary, "worktree", "add", "-q", "-b", "other", str(other))
    return Rig(base=base, home=home, origin=origin, primary=primary, feat=feat, other=other)


def point_session_at(session_dir: Path, cwd: Path) -> None:
    """Give a session a recorded cwd the way a created session carries one."""
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "desktop.json").write_text(json.dumps({"cwd": str(cwd)}), encoding="utf-8")

"""Recoverable, private Git snapshots for children that may die without settling.

A checkpoint is a commit under ``refs/lop/wip`` rather than a stash: stashing
rewrites the shared worktree and index, potentially destroying another lane's
work. This deliberately captures the whole repository, not just files attributed
to the child (Git cannot reliably attribute concurrent writes). Recovery is
explicit: inspect the checkpoint before applying ``restore_hint()``. No recovery
command is executed automatically. Only stdlib/subprocess imports belong here.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def restore_hint(root: str | Path, commit: str) -> str:
    """An explicit overlay recovery command, including previously untracked files.

    Inspect deletions separately against the saved HEAD: an overlay never deletes
    somebody else's concurrent work. Callers must not execute this automatically.
    """
    quoted_root = shlex.quote(str(root))
    return f"git -C {quoted_root} archive {shlex.quote(commit)} | tar -x -C {quoted_root}"


class WipCheckpointer:
    """Keep committed worktree trees without moving HEAD or the real index.

    All Git writes use a temporary index and a dedicated identity. Errors are
    receipts, never child failures; a missing Git installation is ordinary on
    hosts that are not coding machines. The ref keeps otherwise unreachable
    objects alive across process death and garbage collection.
    """

    def __init__(
        self,
        cwd: str | Path,
        child_dir: str | Path,
        job_id: str,
        label: str,
        interval_s: float = 30.0,
    ) -> None:
        self.cwd = Path(cwd)
        self.child_dir = Path(child_dir)
        self.job_id = job_id
        self.label = label
        self.interval_s = interval_s
        self.child_session_id = self.child_dir.name
        self.ref = f"refs/lop/wip/{self.child_session_id}/{job_id}"
        self.path = self.child_dir / f"subagent-wip-{job_id}.v1.json"
        self.root: Path | None = None
        self.result: dict[str, Any] = {}
        self.step = ""
        self.step_at = 0.0
        self.last_snapshot = 0.0
        self.last_receipt = 0.0
        self.number = 0
        try:
            saved = json.loads(self.path.read_text())
            if (
                isinstance(saved, dict)
                and saved.get("job_id") == job_id
                and saved.get("artifact") == "subagent-wip"
            ):
                self.result = saved
                self.step = str(saved.get("step") or "")
                self.step_at = float(saved.get("last_step_at") or 0)
                self.root = Path(saved["root"]) if saved.get("root") else None
        except (OSError, ValueError, TypeError):
            pass

    @staticmethod
    def _environment() -> dict[str, str]:
        # Routing inherited from a launching Git process must not redirect the
        # snapshot to another repository or to somebody else's alternate index.
        return {
            key: value
            for key, value in os.environ.items()
            if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"}
        }

    def _git(self, *args: str, env: dict[str, str] | None = None) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root or self.cwd), *args],
            env=env if env is not None else self._environment(),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()

    def _write(self) -> None:
        payload = {
            "version": 1,
            "artifact": "subagent-wip",
            "job_id": self.job_id,
            "label": self.label,
            "ref": self.ref,
            "root": str(self.root or self.cwd),
            "child_session_id": self.child_session_id,
            "commit": "",
            "tree": "",
            "head": "",
            "files": [],
            "at": time.time(),
            **self.result,
            "step": self.step,
            "last_step_at": self.step_at,
        }
        fd, name = tempfile.mkstemp(dir=self.child_dir, prefix=f".{self.path.name}.")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
            os.chmod(name, 0o600)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.last_receipt = time.monotonic()

    def snapshot(self, force: bool = False) -> dict[str, Any]:
        """Return checkpoint metadata or a diagnostic ``skipped`` receipt.

        Force bypasses only the time debounce. Comparing tree identities avoids
        empty commits even when the worktree remains dirty relative to HEAD.
        """
        if (
            not force
            and self.last_snapshot
            and time.monotonic() - self.last_snapshot < self.interval_s
        ):
            return self.result
        self.last_snapshot = time.monotonic()
        index = ""
        try:
            if self._git("rev-parse", "--is-bare-repository") == "true":
                raise ValueError("bare repository")
            self.root = Path(self._git("rev-parse", "--show-toplevel"))
            try:
                head = self._git("rev-parse", "--verify", "HEAD")
            except subprocess.CalledProcessError:
                head = ""
            fd, index = tempfile.mkstemp(prefix="lop-wip-index-", dir=self.child_dir)
            os.close(fd)
            os.unlink(index)  # read-tree requires an absent or valid index.
            env = self._environment()
            # An inherited index must never route writes into the caller's index.
            env["GIT_INDEX_FILE"] = index
            for role in ("AUTHOR", "COMMITTER"):
                env[f"GIT_{role}_NAME"] = "Local Operator WIP"
                env[f"GIT_{role}_EMAIL"] = "local-operator-wip@localhost"
            self._git("read-tree", head or "--empty", env=env)
            self._git("add", "-A", "--", ".", env=env)
            tree = self._git("write-tree", env=env)
            clean = bool(head) and tree == self._git("rev-parse", f"{head}^{{tree}}")
            if tree == self.result.get("tree") or clean:
                if not self.result.get("commit"):
                    self.result = {"skipped": "clean worktree", "at": time.time()}
                self._write()
                return self.result
            self.number += 1
            parents = ["-p", head] if head else []
            commit = self._git(
                "commit-tree",
                tree,
                *parents,
                "-m",
                f"wip(subagent): {self.label} {self.job_id} checkpoint {self.number}",
                env=env,
            )
            self._git("update-ref", self.ref, commit)
            files = self._git(
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                *([head, commit] if head else ["--root", commit]),
            )
            self.result = {
                "commit": commit,
                "tree": tree,
                "head": head,
                "files": files.splitlines(),
                "at": time.time(),
            }
            self._write()
        except Exception as exc:  # Checkpointing must never end a child run.
            logger.warning("WIP checkpoint skipped: %s", exc)
            self.result = {**self.result, "skipped": str(exc), "at": time.time()}
            try:
                self._write()
            except OSError:
                logger.warning("Could not write WIP skip receipt", exc_info=True)
        finally:
            if index and os.path.exists(index):
                try:
                    os.unlink(index)
                except OSError:
                    logger.warning("Could not remove temporary WIP index", exc_info=True)
        return self.result

    def note_step(self, tool_name: str, intent: str, at: float) -> None:
        """Remember a completed tool, throttling durable receipt churn to 5s."""
        self.step = f"{tool_name}: {intent}" if intent else tool_name
        self.step_at = at
        if time.monotonic() - self.last_receipt >= 5:
            try:
                self._write()
            except OSError:
                logger.warning("Could not write WIP step", exc_info=True)

    def restore_hint(self) -> str:
        """Restore tracked AND untracked archived files, never auto-apply them.

        Archive overlays files; inspect deletions separately against the saved
        HEAD before recovery. This avoids deleting concurrent work on recovery.
        """
        return restore_hint(self.root or self.cwd, str(self.result.get("commit") or self.ref))

    def clear(self) -> None:
        """Release successful work's recovery ref/receipt, best-effort."""
        try:
            self._git("update-ref", "-d", self.ref)
        except (OSError, subprocess.SubprocessError):
            logger.warning("Could not clear WIP ref", exc_info=True)
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not clear WIP receipt", exc_info=True)

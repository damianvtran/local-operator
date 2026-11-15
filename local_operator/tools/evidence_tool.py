"""``evidence``: the sanctioned way a review-type child files its OWN receipt.

WHY THIS TOOL EXISTS
--------------------
Repositories mandate where review and audit receipts live (``i18n/receipts/``,
``i18n/audits/evidence/``, ``docs/evidence/`` — each repository uses its own
subset; see ``EVIDENCE_DIRS``), and review-type roles are
expected to leave one there. The packaged ``reviewer`` row has no
``write``/``edit``; its ``bash`` could physically write the file, but a shell
redirect into a repository is exactly what a careful reader (rightly wary of a
guard-refused reroute) will not do, so it had no SANCTIONED way to put a freshly
authored receipt where the repository requires it.
In session 4e70660bd1bf (local-operator-ui PR #1813) an auditor, after a correct
guard refusal to import ANOTHER session's receipt, also declined to write its
own new receipt for want of any sanctioned path, and the operator had to step
in. This module is that sanctioned path, and nothing wider.

WHAT IT IS, AND WHAT IT IS NOT
------------------------------
It SANCTIONS A PATH; it does not sandbox the reviewer's shell. ``bash`` keeps
exactly the reach it already had (unconfined for ``reviewer`` and the
``tools: null`` reader rows; only ``scout`` is write-confined). What this adds is
one narrow, auditable, named write a reader can use without improvising:

- **Worktree-bound.** The root is fixed when the child is built, from the
  brief's ``Worktree:`` field, and accepted only when that directory is the git
  top level of a checkout git itself lists (primary or linked worktree). The
  model never names the root, so it cannot aim the tool at another tree.
- **Create-only.** ``O_CREAT|O_EXCL|O_NOFOLLOW``: an existing file is never
  replaced. A receipt is a record; overwriting one (or someone else's) would
  turn the sanctioned path into the forgery the guard exists to stop.
- **Content-only.** The tool takes the bytes, never a source path, so it cannot
  copy another session's material into the tree by reference. What the child
  writes is what it authored this turn (``guide://guardrails``: fresh work, not
  a reroute).
- **Allow-listed directories that already exist.** Only under the directories
  in :data:`EVIDENCE_DIRS`, and only where that directory already exists in the
  bound checkout (local-operator has no ``i18n/receipts/``; the UI does).
  Subdirectories below it may be created; every component is opened
  ``O_NOFOLLOW`` relative to its parent's descriptor, so no symlink, ``..``,
  ``~`` or absolute spelling can leave the tree, including by a race.
- **Guard evidence stays CRITICAL.** The resolved target is also checked with
  :func:`local_operator.guardrails.check.evidence_write_verdict`, and the tool
  is registered in ``FILE_DESTINATION_TOOLS`` so the checkpoint ranks it too.

Granted only to delegated review-type children whose brief names a valid
worktree (``harness/subagent.py``, :func:`reader_evidence_root`); never to a
top-level session, a coder or a manager. Model-facing text here (description,
refusals) is English by design: AGENTS.md "Internationalization is mandatory"
exempts model-facing prompts.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import subprocess
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import AgentTool, ToolResult
from local_operator.tools.builtin import _error, _text, _validation_error

TOOL_NAME = "evidence"

#: Repo-mandated receipt locations, relative to the checkout root. This is a
#: CROSS-REPOSITORY allowlist, not a statement that every checkout has all three.
#: Verified on origin/main 2026-11-11: local-operator-ui has all three;
#: local-operator tracks only ``i18n/audits/evidence`` — its ``docs/evidence``
#: was retired by the repository cleanup (PR #3683; ``/docs/evidence/`` is now
#: in its ``.gitignore``, and AGENTS.md "Testing evidence must not be committed"
#: sends proof captures to the PR instead). ``docs/evidence`` stays listed for
#: local-operator-ui. The tool refuses a directory the bound checkout lacks
#: rather than creating it, so listing one here never invents a location a
#: repository did not choose; a clean local-operator worktree has no
#: ``docs/evidence`` and therefore refuses that prefix.
EVIDENCE_DIRS: tuple[str, ...] = ("i18n/receipts", "i18n/audits/evidence", "docs/evidence")

#: Content cap in UTF-8 bytes. Measured 2026-11-11 over the tracked files in
#: those directories (both repos, origin/main): the largest text receipt/audit
#: file is 540,949 bytes (UI ``i18n/audits/evidence``), UI ``i18n/receipts``
#: max 169,313, p99 of every directory <= 660,938. 1 MiB is ~1.9x the largest
#: i18n file and above every p99. The one larger outlier (a 6.4 MB UI
#: ``docs/evidence`` capture) is a measurement dump, not a receipt a reviewer
#: authors in one call; it belongs on the PR (AGENTS.md "Evidence goes on the
#: PR").
MAX_EVIDENCE_BYTES = 1 << 20

_GIT_TIMEOUT_S = 10.0


class EvidenceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Relative path under an allowed dir, e.g. i18n/receipts/x.json")
    content: str = Field(description="Full file text you authored")


def _git(root: str, *argv: str) -> str | None:
    # No credential helper or prompt can be reached by these read-only
    # plumbing calls, but pin it anyway: this runs at child build, unattended.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
    try:
        done = subprocess.run(
            ["git", "-C", root, *argv],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def verified_worktree_root(path_text: str) -> str | None:
    """``path_text``'s realpath when it is a git checkout's own top level that
    git lists as a worktree (the primary checkout counts), else None.

    A plain directory, a subdirectory of a checkout, or a stray ``.git`` file
    pointing at nothing are all refused: the binding must name a tree the
    operator's git knows, so the brief cannot point the tool at an arbitrary
    directory just by writing its path.
    """
    expanded = os.path.expanduser(path_text.strip())
    if not expanded or not os.path.isabs(expanded) or not os.path.isdir(expanded):
        return None
    real = os.path.realpath(expanded)
    top = _git(real, "rev-parse", "--show-toplevel")
    if top is None or os.path.realpath(top.strip()) != real:
        return None
    listing = _git(real, "worktree", "list", "--porcelain")
    if listing is None:
        return None
    for line in listing.splitlines():
        if line.startswith("worktree ") and os.path.realpath(line[len("worktree ") :]) == real:
            return real
    return None


def first_verified_root(candidates: Iterable[str]) -> str | None:
    """The first candidate that :func:`verified_worktree_root` accepts.

    First VALID, not first named: a role preamble or team brief can quote the
    field as a template (``Worktree: <abs path> @ <sha>``) ahead of the
    manager's real line.
    """
    for candidate in candidates:
        root = verified_worktree_root(candidate)
        if root is not None:
            return root
    return None


class EvidenceRefused(Exception):
    """A rule refused the write; ``str(exc)`` is the agent-facing reason."""


def _split_relative(path: str) -> list[str]:
    if not path or not path.strip():
        raise EvidenceRefused("path is empty; give a relative path such as i18n/receipts/<name>.")
    if "\x00" in path or "\\" in path:
        raise EvidenceRefused("path may not contain NUL or backslash characters.")
    if path.startswith("~"):
        raise EvidenceRefused("`~` paths are refused; give a path relative to the worktree root.")
    if os.path.isabs(path):
        raise EvidenceRefused(
            "absolute paths are refused; give a path relative to the worktree root."
        )
    parts = path.split("/")
    if any(part == ".." for part in parts):
        raise EvidenceRefused("`..` components are refused; the file must stay in the worktree.")
    if any(part in ("", ".") for part in parts):
        raise EvidenceRefused("empty or `.` path components are refused; write the path plainly.")
    return parts


def _allowed_split(parts: list[str]) -> tuple[list[str], list[str]]:
    for allowed in EVIDENCE_DIRS:
        prefix = allowed.split("/")
        if parts[: len(prefix)] == prefix and len(parts) > len(prefix):
            return prefix, parts[len(prefix) :]
    raise EvidenceRefused(
        "path is outside the evidence directories; allowed: "
        + ", ".join(f"{d}/" for d in EVIDENCE_DIRS)
        + "."
    )


def _open_dir(name: str, parent_fd: int, *, create: bool, shown: str) -> int:
    """Open ``name`` under ``parent_fd`` as a real directory, never a link."""
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            raise EvidenceRefused(
                f"{shown}/ does not exist in this worktree; this repository does "
                "not use that evidence directory."
            ) from None
        try:
            os.mkdir(name, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode):
        raise EvidenceRefused(f"{shown} is a symlink; symlinked path components are refused.")
    if not stat.S_ISDIR(info.st_mode):
        raise EvidenceRefused(f"{shown} is not a directory.")
    try:
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        # Swapped for a link between the stat and the open: same rule.
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise EvidenceRefused(
                f"{shown} is a symlink; symlinked path components are refused."
            ) from None
        raise


def _guard_refusal(target: str) -> str | None:
    from local_operator.guardrails.check import CallContext, evidence_write_verdict

    ctx = CallContext(cwd=None, env=dict(os.environ), scratchpad=None)
    verdict = evidence_write_verdict(TOOL_NAME, {"path": target}, ctx)
    if verdict is None:
        return None
    reason = verdict.targets[0].reason if verdict.targets else "guardrail evidence"
    return f"{target} is {reason}; the guardrail's evidence is never writable."


def write_evidence(root: str, path: str, content: str) -> tuple[str, int, str]:
    """Create ``root/path`` with ``content``; return (path, bytes, sha256).

    Raises :class:`EvidenceRefused` naming the rule on any refusal.
    """
    data = content.encode("utf-8")
    if len(data) > MAX_EVIDENCE_BYTES:
        raise EvidenceRefused(
            f"content is {len(data)} bytes; the evidence cap is {MAX_EVIDENCE_BYTES} "
            "bytes. Attach larger captures to the PR instead."
        )
    parts = _split_relative(path)
    allowed, below = _allowed_split(parts)
    target = os.path.join(root, *parts)
    refusal = _guard_refusal(target)
    if refusal is not None:
        raise EvidenceRefused(refusal)
    fds: list[int] = []
    try:
        fds.append(os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        walked: list[str] = []
        # The allow-listed directory itself must already exist; only the
        # subdirectories under it may be created.
        for index, name in enumerate([*allowed, *below[:-1]]):
            walked.append(name)
            fds.append(
                _open_dir(name, fds[-1], create=index >= len(allowed), shown="/".join(walked))
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            fd = os.open(below[-1], flags, 0o644, dir_fd=fds[-1])
        except FileExistsError:
            raise EvidenceRefused(
                f"{path} already exists; evidence is create-only and never overwrites. "
                "Choose a new file name for your own receipt."
            ) from None
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise EvidenceRefused(f"{path} is a symlink; refused.") from None
            raise
        try:
            os.fchmod(fd, 0o644)  # the umask must not decide a receipt's mode
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
        finally:
            os.close(fd)
    finally:
        for fd_open in reversed(fds):
            os.close(fd_open)
    return path, len(data), hashlib.sha256(data).hexdigest()


def build_evidence_tool(root: str) -> AgentTool:
    """The ``evidence`` tool bound to the verified worktree ``root``."""

    async def execute(tool_call_id, args, signal=None, on_update=None, context=None) -> ToolResult:
        try:
            params = EvidenceParams(**args)
        except ValidationError as exc:
            return _validation_error(tool_call_id, TOOL_NAME, exc)
        try:
            written, size, digest = write_evidence(root, params.path, params.content)
        except EvidenceRefused as exc:
            return _error(tool_call_id, TOOL_NAME, f"evidence refused: {exc}")
        except OSError as exc:
            return _error(tool_call_id, TOOL_NAME, f"evidence write failed: {exc.strerror}")
        return _text(
            tool_call_id,
            TOOL_NAME,
            # A receipt lands untracked in the shared PR worktree and readers
            # cannot commit, so superseded rounds were left as `??` after merge.
            # Saying who commits it is what keeps the merged tree status-clean.
            f"Created {written} in {root} ({size} bytes, sha256 {digest}). "
            "Untracked until the PR owner commits it: name this path in your report.",
            details={"root": root, "path": written, "bytes": size, "sha256": digest},
        )

    return AgentTool(
        name=TOOL_NAME,
        label="Write evidence",
        description=(
            "Create a NEW receipt file you authored in the briefed worktree's "
            "i18n/receipts, i18n/audits/evidence or docs/evidence (only the ones "
            "that repository already has). Never overwrites."
        ),
        parameters=EvidenceParams.model_json_schema(),
        approval_tier="write",
        concurrency="exclusive",
        interruptible=False,
        execute=execute,
    )

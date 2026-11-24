"""Read-only repository trees for a responder run, without running repo code (S-12).

UNWIRED BY DESIGN (see ``jail.py``).

WHAT. A responder run reads source it was allowed to (``allowed_repos``), but
the jail denies the rest of ``$HOME`` -- including the owner's clones. So the
unjailed host copies each allowed repository's ``origin/main`` tree into
``<run home>/repos/<name>`` before the spawn, read-only.

WHY NOT ``git archive`` (security design round 2, S-12, measured there on git
2.55.0): ``git archive`` runs the repository-config ``filter.<x>.smudge``
driver, whether the attribute comes from the tree's ``.gitattributes`` or only
from ``.git/info/attributes``. A same-uid planter who writes both into the
owner's clone would get a command run on EVERY alert wake, outside the jail,
with the owner's environment and SSO session. This module therefore reads
objects RAW: ``git ls-tree -r -z`` for the listing and ``git cat-file --batch``
for the bytes. Neither applies attributes, filters or textconv, and neither
touches the index or the worktree (so ``core.fsmonitor`` never runs); hooks are
pointed at ``/dev/null`` and system/global config are not read at all, as a
belt to that brace.

PINNED SOURCE. Only ``radient-ml/<name>`` is materialized, from
``~/radient-ml/<name>`` (design §4: no GitHub token, no other org). Anything
else raises :class:`RepoNotAllowed`; C2 uses :func:`pinned_repo_name` at policy
parse to answer ``policy_invalid``.

WHAT IS SKIPPED. Symlinks (mode 120000) -- a link materialized into the run
home could point anywhere; the jail would still refuse the read, but there is no
reason to hand the run a pointer at the owner's tree -- and gitlinks (160000,
submodules: no object here to copy). Any path with a ``.git`` component is
skipped so no nested repository configuration lands where a jailed ``git``
would honour it.

RESIDUAL (design S-12, accepted): a planter can repoint the clone's
``refs/remotes/origin/main`` to serve the run another tree. The commit SHA is
recorded in the isolation attestation so the tree a run saw is auditable.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

#: The one GitHub org whose repositories a responder may read (design §4).
RADIENT_ORG = "radient-ml"

#: The ref materialized: what the owner last fetched from GitHub's ``main``.
MATERIALIZED_REF = "refs/remotes/origin/main"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_SHA = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")

#: ``-c`` overrides on every git call: no hooks, no fsmonitor daemon.
_GIT_OVERRIDES: tuple[str, ...] = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false")

#: The git environment: rebuilt, never the owner's. No system or global
#: config (Apple Git reads a second system file that ``GIT_CONFIG_SYSTEM``
#: alone does not suppress, hence both), no prompts, no attribute files beyond
#: the repository (and the raw plumbing used here reads none anyway).
_GIT_ENV: dict[str, str] = {
    "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "/bin/false",
    "GIT_OPTIONAL_LOCKS": "0",
    "LC_ALL": "C",
}


class RepoNotAllowed(ValueError):
    """An ``allowed_repos`` entry outside ``radient-ml/<name>``."""


class MaterializeError(RuntimeError):
    """The tree could not be copied; the spawner answers ``spawn_failed``."""


@dataclass(frozen=True)
class MaterializedRepo:
    """One copied tree: what the attestation records."""

    entry: str  # "radient-ml/<name>"
    commit: str
    path: Path
    files: int
    skipped: int


def pinned_repo_name(entry: str) -> str:
    """``radient-ml/<name>`` -> ``<name>``; anything else raises :class:`RepoNotAllowed`."""

    org, sep, name = entry.partition("/")
    if org != RADIENT_ORG or not sep or _NAME.fullmatch(name) is None or name in {".", ".."}:
        raise RepoNotAllowed(f"{entry!r} is not a {RADIENT_ORG}/<name> repository")
    return name


def default_clone_root() -> Path:
    """Where the owner's sibling clones live (``~/radient-ml``)."""

    return Path.home() / RADIENT_ORG


def _git(clone: Path, *args: str, stdin: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(clone), *_GIT_OVERRIDES, *args],
        input=stdin,
        capture_output=True,
        env=_GIT_ENV,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip().splitlines()[-1:] or [
            "no detail"
        ]
        raise MaterializeError(f"git {args[0]} failed in {clone.name}: {detail[0]}")
    return completed.stdout


def _safe_relpath(raw: bytes) -> PurePosixPath | None:
    """The tree path if it is a plain relative path without ``.git``, else ``None``."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts:
        return None
    if any(part in {"", ".", ".."} or part.lower() == ".git" for part in path.parts):
        return None
    return path


def materialize_repo(
    entry: str,
    dest_root: Path,
    *,
    clone_root: Path | None = None,
) -> MaterializedRepo:
    """Copy ``entry``'s :data:`MATERIALIZED_REF` tree into ``dest_root/<name>``, read-only.

    ``dest_root`` is the run home's ``repos/`` directory; the destination must
    not exist (a run home is fresh per alert). Raises :class:`RepoNotAllowed`
    or :class:`MaterializeError`; never leaves a partial tree behind.
    """

    name = pinned_repo_name(entry)
    clone = (clone_root or default_clone_root()) / name
    if not clone.is_dir():
        raise MaterializeError(f"no local clone of {entry} at {clone}")
    dest = dest_root / name
    if dest.exists() or dest.is_symlink():
        raise MaterializeError(f"{dest} already exists")

    commit = _git(
        clone, "rev-parse", "--verify", "--end-of-options", f"{MATERIALIZED_REF}^{{commit}}"
    )
    commit_sha = commit.decode().strip()
    if _SHA.fullmatch(commit_sha) is None:
        raise MaterializeError(f"{entry}: unexpected rev-parse output")

    listing = _git(clone, "ls-tree", "-r", "-z", "--full-tree", commit_sha)
    blobs: list[tuple[PurePosixPath, str, str]] = []
    skipped = 0
    for record in filter(None, listing.split(b"\0")):
        meta, _, raw_path = record.partition(b"\t")
        mode, kind, sha = meta.decode().split(" ")
        path = _safe_relpath(raw_path)
        if kind != "blob" or mode not in {"100644", "100755"} or path is None:
            skipped += 1
            continue
        blobs.append((path, mode, sha))

    staging = dest_root / f".{name}.partial"
    if staging.exists():
        _make_writable_and_remove(staging)
    staging.mkdir(parents=True)
    try:
        payload = _git(
            clone, "cat-file", "--batch", stdin="".join(f"{sha}\n" for _, _, sha in blobs).encode()
        )
        offset = 0
        for path, mode, sha in blobs:
            header_end = payload.index(b"\n", offset)
            got_sha, kind, size = payload[offset:header_end].decode().split(" ")
            if got_sha != sha or kind != "blob":
                raise MaterializeError(f"{entry}: object stream out of order at {path}")
            start = header_end + 1
            end = start + int(size)
            target = staging.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload[start:end])
            target.chmod(0o555 if mode == "100755" else 0o444)
            offset = end + 1  # the LF cat-file writes after each object
        for directory, _dirs, _files in os.walk(staging, topdown=False):
            os.chmod(directory, 0o555)
        os.chmod(staging, 0o755)
        staging.rename(dest)
        os.chmod(dest, 0o555)
    except BaseException:
        _make_writable_and_remove(staging)
        raise
    return MaterializedRepo(
        entry=entry, commit=commit_sha, path=dest, files=len(blobs), skipped=skipped
    )


def _make_writable_and_remove(tree: Path) -> None:
    """Remove a staging tree this module created (read-only dirs need u+w first)."""

    for directory, _dirs, _files in os.walk(tree):
        os.chmod(directory, stat.S_IRWXU)
    shutil.rmtree(tree, ignore_errors=True)


__all__ = [
    "MATERIALIZED_REF",
    "RADIENT_ORG",
    "MaterializeError",
    "MaterializedRepo",
    "RepoNotAllowed",
    "default_clone_root",
    "materialize_repo",
    "pinned_repo_name",
]

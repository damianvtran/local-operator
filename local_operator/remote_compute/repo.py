"""Repo and commit for a run, taken from a working directory — or refused with a reason.

WHAT A REMOTE RUN NEEDS, AND WHY THIS IS NOT ``--repo`` WITH DEFAULTS. The far
end clones a repository and checks out ONE COMMIT, so the two facts it needs are
the repository's URL and the 40-character SHA. A desktop panel cannot make the
operator retype them, so it reads them from the directory the operator is
pointing at — and that read is where two silent failures live:

* **An SSH origin is not cloneable by the container.** ``git@host:path`` needs a
  key and an agent, and the pod has neither. Passing it on produced a run that
  failed minutes later with a checkout error about a remote nobody could see.
* **An unpushed commit does not exist on the remote.** The container clones what
  the server holds, so a run whose ``repo_sha`` was committed on this machine and
  never pushed fails at the far end — or worse, clones a different commit if the
  SHA happens to exist there. It is refused here, before a request is composed.

So detection either answers a URL and a SHA the container can actually fetch, or
it raises :class:`~local_operator.remote_compute.errors.UnusableRepo` with a
reason the surface has copy for.

WHAT "PUSHED" MEANS HERE: present on a remote-tracking ref of ``origin``, as this
clone last observed it. That is deliberately a LOCAL fact — no network read
happens unless the caller asks for one with ``refresh=True``. A push from this
clone updates the tracking ref, so the ordinary path needs no fetch; a push made
from another machine does not, and refusing it is the safe direction (the
remedy, ``git fetch``, is in the copy).
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import urllib.parse
from pathlib import Path

from local_operator.remote_compute.errors import UnusableRepo
from local_operator.remote_compute.request import COMMIT_SHA_PATTERN

#: The remote a run is submitted from. One name, because the URL that is sent is
#: ``origin``'s and the pushed-check is about ``origin``'s tracking refs: a run
#: assembled from one remote and verified against another is the disagreement
#: this constant prevents.
REMOTE_NAME = "origin"

#: How long a local git read may take. Generous for a large repository on a cold
#: cache, short enough that a panel does not hang on a wedged index.
GIT_TIMEOUT_SECONDS = 10.0

#: A network fetch may take longer; it is opt-in (:func:`detect_repo`).
GIT_FETCH_TIMEOUT_SECONDS = 60.0

#: The reasons :class:`UnusableRepo` may carry, as this module produces them.
REASON_NOT_HTTPS = "not_https_remote"
REASON_UNPUSHED = "unpushed_commit"
REASON_NO_REMOTE = "no_remote"


@dataclasses.dataclass(frozen=True)
class RepoRef:
    """A repository the container can clone, and the commit it will check out."""

    directory: Path
    #: The ``origin`` URL, https and credential-free — exactly what is sent.
    url: str
    #: The 40-character commit, as ``git rev-parse HEAD`` answered.
    sha: str
    #: The ``origin`` tracking refs that contain ``sha``, for a receipt or a log.
    #: Non-empty by construction: an empty answer is the unpushed refusal.
    refs: tuple[str, ...]


def detect_repo(directory: Path | str, *, refresh: bool = False) -> RepoRef:
    """The repository and HEAD commit of ``directory``, or a typed refusal.

    ``refresh`` runs ``git fetch origin`` first. It is OFF by default because a
    detection helper that reaches the network is a detection helper that blocks:
    a panel opening on an offline laptop must still be able to say what it knows.
    """
    root = _repository_root(Path(directory))
    if refresh:
        _git(root, "fetch", "--quiet", REMOTE_NAME, timeout=GIT_FETCH_TIMEOUT_SECONDS)
    url = _https_origin(root)
    sha = _head_sha(root)
    refs = _containing_refs(root, sha)
    if not refs:
        raise UnusableRepo(
            REASON_UNPUSHED,
            detail=f"{sha[:12]} ({_head_branch(root) or 'HEAD'})",
        )
    return RepoRef(directory=root, url=url, sha=sha, refs=refs)


def _repository_root(directory: Path) -> Path:
    """The directory's repository root, or ``no_remote``.

    ``--show-toplevel`` rather than a ``.git`` existence test: a subdirectory of a
    checkout, a linked worktree and a bare-ish layout all answer correctly, and a
    directory that is merely INSIDE a repository is still the repository the
    operator meant.
    """
    if not directory.is_dir():
        raise UnusableRepo(REASON_NO_REMOTE, detail=str(directory))
    code, output = _git(directory, "rev-parse", "--show-toplevel")
    if code != 0 or not output:
        raise UnusableRepo(REASON_NO_REMOTE, detail=str(directory))
    return Path(output)


def _https_origin(root: Path) -> str:
    """``origin``'s URL, refused unless it is https and carries no credentials.

    THE URL IS STORED AND SENT. It goes into the run body and into this device's
    local run record, so a URL that embeds a username and password would put a
    credential on disk and in a request body; the container cannot use one
    anyway. Both refusals share a reason because they share a remedy: point
    ``origin`` at a credential-free https URL.
    """
    code, output = _git(root, "remote", "get-url", REMOTE_NAME)
    if code != 0 or not output:
        raise UnusableRepo(REASON_NO_REMOTE, detail=str(root))
    raw = output.strip()
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme != "https" or "@" in parts.netloc or not parts.netloc:
        raise UnusableRepo(REASON_NOT_HTTPS, detail=scrub_url(raw))
    return raw


def _head_sha(root: Path) -> str:
    """``HEAD``'s commit, refused unless it is a 40-character SHA.

    ``rev-parse HEAD`` fails on a repository with no commits, and a symbolic
    answer (an unborn branch) is not a commit; both are ``no_remote`` — there is
    no commit here for the container to check out.
    """
    code, output = _git(root, "rev-parse", "HEAD")
    sha = output.strip()
    if code != 0 or COMMIT_SHA_PATTERN.fullmatch(sha) is None:
        raise UnusableRepo(REASON_NO_REMOTE, detail=str(root))
    return sha


def _head_branch(root: Path) -> str:
    """``HEAD``'s short branch name, or ``""`` when it is detached."""
    code, output = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if code != 0 or not output or output == "HEAD":
        return ""
    return output


def _containing_refs(root: Path, sha: str) -> tuple[str, ...]:
    """``origin``'s tracking refs that contain ``sha``, as this clone knows them.

    ``--list 'origin/*'`` restricts the answer to the remote the run is submitted
    from: a commit pushed to a fork is not on ``origin``, and a run built from
    the fork's tracking ref would send a SHA the container cannot fetch.
    """
    code, output = _git(root, "branch", "-r", "--contains", sha, "--list", f"{REMOTE_NAME}/*")
    if code != 0:
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _git_environment() -> dict[str, str]:
    """The environment every git read here runs with: no ``GIT_*`` at all.

    ``git -C <directory>`` IS NOT AUTHORITATIVE while the parent shell carries a
    repository-location variable: with ``GIT_DIR`` or ``GIT_WORK_TREE`` exported
    (a shell that then started the panel), git answers about THAT repository and
    this module reads another checkout's origin URL and another commit's pushed
    state — a run submitted, under the operator's identity, as the wrong code.
    Dropping every inherited ``GIT_*`` rather than a named list is the same rule
    ``guardrails/provenance._git_proc`` settled on, and it also takes the
    inherited credential helpers and config overrides with it.

    The two variables added back are the read-only ones: no prompt (a panel has no
    terminal to answer one, and a hang there is indistinguishable from a slow
    network) and no index refresh writes (``git status``-class reads must not
    touch the repository they are inspecting).
    """
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    return environment


def _git(directory: Path, *arguments: str, timeout: float = GIT_TIMEOUT_SECONDS) -> tuple[int, str]:
    """Run one git command in ``directory``; ``(returncode, stdout)``.

    STDOUT IS STRIPPED HERE, once, so no caller has to remember that git ends
    its output with a newline — a trailing one in a path or a SHA is the kind of
    defect that only shows up as "the repository is not a repository".

    NO SHELL, and :func:`_git_environment` for the environment, so a fetch that
    wants credentials fails instead of waiting on a terminal and an inherited
    ``GIT_DIR`` cannot point the answer at another repository. A missing ``git``
    binary is reported as ``(127, "")``: the caller's ordinary refusal path then
    says the directory is not a repository, which is what an install without git
    can honestly say.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    return completed.returncode, completed.stdout.strip()


def scrub_url(url: str) -> str:
    """A URL safe to store or carry in a refusal: ``scheme://host/path``, no userinfo.

    The URL reaches a log, a panel and — via :func:`record_submission` — this
    device's run record, so the userinfo segment, where a mistaken
    ``https://user:token@…`` puts a credential, is dropped rather than kept. The
    host and path stay, because they are what the operator needs in order to find
    the remote they configured wrongly.

    THIS NEVER RAISES. ``urlsplit().port`` raises ``ValueError`` for an
    out-of-range port (``https://example.com:99999999/r.git``, one keystroke away
    from a real one in ``git remote set-url``), and both callers promise a typed
    answer: ``detect_repo`` returns ``RepoRef | UnusableRepo`` and
    ``record_submission`` returns a record. So a malformed port falls back to the
    netloc with its userinfo removed, which still shows the operator the value
    they typed.
    """
    parts = urllib.parse.urlsplit(url)
    if not parts.scheme:
        return re.sub(r"^[^/@]*@", "", url)
    try:
        port = parts.port
    except ValueError:
        return f"{parts.scheme}://{_without_userinfo(parts.netloc)}{parts.path}"
    host = parts.hostname or ""
    if port is not None:
        host = f"{host}:{port}"
    if not host:
        return f"{parts.scheme}://"
    return f"{parts.scheme}://{host}{parts.path}"


def _without_userinfo(netloc: str) -> str:
    """``netloc`` minus any ``user:password@`` prefix.

    ``rpartition`` rather than ``partition``: a password may itself contain ``@``
    and the split has to be at the LAST one, which is where the host begins.
    """
    return netloc.rpartition("@")[2]


__all__ = [
    "GIT_FETCH_TIMEOUT_SECONDS",
    "GIT_TIMEOUT_SECONDS",
    "REASON_NO_REMOTE",
    "REASON_NOT_HTTPS",
    "REASON_UNPUSHED",
    "REMOTE_NAME",
    "RepoRef",
    "detect_repo",
    "scrub_url",
]

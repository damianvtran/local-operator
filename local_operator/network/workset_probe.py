"""Can the DESTINATION use the moved worktrees? Asked of the destination itself.

WHY THIS EXISTS (live Mac -> node2 move, 2026-11-18 17:12Z). The manifest said
``clone_check: "pending_destination"`` for every worktree and the user confirmed it;
on arrival node2 could not fetch the SSH remote at all and could not push to the
HTTPS one, so the work was unusable there even once it is transferred (PR B). Only
the destination can answer "can I fetch and push this remote with MY credentials",
so the preflight now asks it, BEFORE the confirmation, over a move-scoped phase
(``net_session_move {phase: "workset_probe"}``) the destination serves.

WHAT THE DESTINATION RUNS, AND WHY THAT IS SAFE:

* Only remote URLs the source sent, and only after :func:`classify_url` accepted
  them: ``https://``, ``ssh://`` and the scp form ``user@host:path``. Everything else
  — ``file://``, bare local paths, ``ext::``/``<helper>::`` transports, an argument
  that starts with ``-``, whitespace/control characters — is refused WITHOUT running
  git (``destination_cannot_clone``, ``detail: "url_refused"``). A hostile source
  must not be able to make the destination run an arbitrary git transport or read
  its local repositories. ``protocol.ext.allow=never`` is passed as a belt.
* ``git ls-remote <url> HEAD`` — the fetch check (read-only by definition).
* ``git push --dry-run <url> <sha>:refs/heads/lop-move-probe-<hex>`` from a
  throwaway local repository holding one empty commit — the push check. CHOSEN OVER
  the GitHub API permissions endpoint because (a) it is host-agnostic (GitLab,
  self-hosted, SSH-only users have no API token), and (b) it exercises the exact
  transport and credential the restore will use (SSH key, credential helper, or the
  mesh-lent ``github`` helper), where an API answer only describes a token. Write
  permission is enforced at receive-pack's ref advertisement — GitHub answers 403
  on ``info/refs?service=git-receive-pack`` over HTTPS and "Permission denied" over
  SSH — and ``--dry-run`` stops before any pack or ref update is sent, so the probe
  performs NO remote write: the branch name is never created.
* Its OWN credentials: this process's environment (SSH agent/keys, configured
  credential helpers) plus, when a ``github.com`` HTTPS URL is involved, the
  mesh-lent ``github`` helper (``credentials.github.borrowed_git_env``) — never
  anything the source sent. Prompts are impossible: ``GIT_TERMINAL_PROMPT=0``,
  stdin from ``/dev/null`` and a new session (no controlling tty for ssh).

FAILURE IS A FLAG, NOT A STATE CHANGE (decision, 2026-11-18). A worktree the
destination cannot fetch or push stays ``planned`` with ``clone_check`` naming the
failure and a ``remedy``; it is NOT turned into ``left_behind``. Two reasons: the
scope digest covers each row's state, and the move gate recomputes the manifest
without re-probing (the probe is network work and its answer can flap), so a
probe-derived state would refuse every confirmed move ``working_set_changed``; and
in PR B a destination that cannot fetch still receives the commits it lacks in the
bundle — it is pushing the work onward that needs the credential. The transfer
half's plan step decides what to leave behind with the probe's answer in hand.

PREREQUISITES ARE WARN-ONLY: missing toolchains (inferred from marker files the
source names, checked with ``shutil.which`` here) and Team Bridge enrolment (only
when the source session shows it uses the bridge) never block a move.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: 2 (review round 1): rows are keyed by ``remote_url`` and carry no path or branch
#: (S6), and the answer carries only booleans and fixed codes (S1/S3).
PROBE_VERSION = 2

#: The ONLY failure details a destination may return (security S1.4, round 1). git's
#: stderr is classed HERE, on the destination, and never leaves it: its first line
#: carried a redirected internal service's error body back to the asker, and its
#: wording ("Failed to connect to <host>:<port>", "Connection closed by ...") made
#: the probe a port scanner. The asker keeps only a value from this tuple.
DETAIL_CODES = (
    "auth_denied",
    "not_found",
    "unreachable",
    "tls",
    "host_key",
    "timeout",
    "host_refused",
    "url_refused",
    "git_missing",
    "probe_error",
    "error",
)

#: Every ``clone_check`` value a block may carry. ``pending_destination`` is the
#: unprobed default (``WorktreeFacts.to_dict``); ``no_remote`` is a worktree with no
#: remote at all (nothing to fetch or push — PR B carries its whole history).
CLONE_CHECKS = (
    "pending_destination",
    "ok",
    "destination_cannot_fetch",
    "destination_cannot_push",
    "destination_cannot_clone",
    "unknown_destination",
    "no_remote",
)

#: Marker file -> toolchains its repository needs on the destination. Exact names at
#: the worktree root only; a marker names a NEED, ``shutil.which`` decides "missing".
TOOLCHAIN_MARKERS: dict[str, tuple[str, ...]] = {
    "go.mod": ("go",),
    "pnpm-lock.yaml": ("pnpm", "node"),
    "package-lock.json": ("npm", "node"),
    "uv.lock": ("uv",),
    "pyproject.toml": ("uv",),
    "Cargo.toml": ("cargo",),
}

#: Bounds. A source may send at most this many rows (a manifest never comes close),
#: each git call is killed after the timeout, and at most four run at once so a
#: probe cannot monopolise a relay.
MAX_PROBE_ROWS = 64
GIT_PROBE_TIMEOUT_S = 20.0
_MAX_WORKERS = 4
#: The asker's wait for the whole answer: two serial git calls per URL, four lanes.
ASK_TIMEOUT_S = 120.0

#: TEST SEAM ONLY: extra ``-c key=value`` pairs for every probe git call (the unit
#: tests map an ``https://git.test/...`` URL onto a local bare repository with
#: ``url.<path>.insteadOf``). Empty in production; never set from a frame.
_EXTRA_CONFIG: tuple[str, ...] = ()

#: TEST SEAM / OPERATOR ALLOWANCE: host names the destination may probe even though
#: they resolve to a non-public address (security S1.3). Empty in production; the
#: destination's operator may name their own self-hosted git servers in
#: ``LOCAL_OPERATOR_MOVE_PROBE_HOSTS`` (comma-separated), which is THIS device's
#: environment, never anything a frame carries.
_ALLOWED_HOSTS: tuple[str, ...] = ()
_ALLOWED_HOSTS_ENV = "LOCAL_OPERATOR_MOVE_PROBE_HOSTS"

_HOST = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,252}[A-Za-z0-9])?$")
#: ssh user names: never option-shaped (``-oProxyCommand=...``), so the allowlist does
#: not lean on git >= 2.14.1's own "strange hostname" refusal (security S7).
_USER = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._-]{0,63}$")
_SCP = re.compile(
    r"^(?:(?P<user>[A-Za-z0-9._][A-Za-z0-9._-]{0,63})@)?"
    r"(?P<host>[A-Za-z0-9.-]{2,253}):(?P<path>[^:]+)$"
)
_UNSAFE_CHARS = re.compile(r"[\s\x00-\x1f\x7f\\]")
#: git's built-in empty tree (always resolvable, even in an empty repository).
_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


# ---------------------------------------------------------------------------
# URL policy
# ---------------------------------------------------------------------------


def classify_url(url: str) -> str | None:
    """``"https"`` / ``"ssh"`` for a URL the destination may probe, else ``None``.

    Allowlist, never a denylist: anything not positively recognised is refused,
    which is what keeps ``file://``, local paths, ``ext::sh -c ...`` and any future
    transport-helper syntax out by construction.
    """
    text = str(url or "")
    if not text or len(text) > 2048 or text.startswith("-") or _UNSAFE_CHARS.search(text):
        return None
    if "::" in text:  # ``<transport>::<address>`` remote-helper syntax
        return None
    if "://" in text:
        try:
            parts = urlsplit(text)
        except ValueError:
            return None
        scheme = parts.scheme.lower()
        if scheme not in ("https", "ssh"):
            return None
        try:
            host = parts.hostname or ""
            parts.port  # noqa: B018 — raises on a malformed port
        except ValueError:
            return None
        if not _HOST.match(host) or not parts.path.strip("/"):
            return None
        if parts.username is not None and not _USER.match(parts.username):
            return None
        return scheme
    match = _SCP.match(text)
    if match is None or not _HOST.match(match.group("host")):
        return None
    # git reads ``a/b:c`` as a local path, never scp; ``_SCP`` already has no slash
    # in the host, and a path starting with ``-`` could be read as an option by ssh.
    if match.group("path").startswith("-"):
        return None
    return "ssh"


def https_rewrite(url: str) -> str:
    """The HTTPS spelling of an SSH remote (``git@host:o/r.git`` -> ``https://host/o/r.git``).

    ``""`` for a URL that is already HTTPS or not an accepted SSH form. Port and user
    are dropped (an SSH port is not the HTTPS port). Only ever OFFERED in a block; PR B
    applies it at restore, and only on this device.
    """
    if classify_url(url) != "ssh":
        return ""
    if "://" in url:
        parts = urlsplit(url)
        host, path = parts.hostname or "", parts.path
    else:
        match = _SCP.match(url)
        if match is None:
            return ""
        host, path = match.group("host"), match.group("path")
    path = path.lstrip("/").lstrip("~").lstrip("/")
    return f"https://{host}/{path}" if host and path else ""


def github_repository(url: str) -> str:
    """``owner/repo`` for a github.com remote in any accepted form, else ``""``."""
    if classify_url(url) is None:
        return ""
    if "://" in url:
        parts = urlsplit(url)
        host, path = (parts.hostname or "").lower(), parts.path
    else:
        match = _SCP.match(url)
        if match is None:
            return ""
        host, path = match.group("host").lower(), match.group("path")
    if host != "github.com":
        return ""
    bits = [bit for bit in path.strip("/").split("/") if bit]
    if len(bits) != 2:
        return ""
    repo = bits[1][:-4] if bits[1].endswith(".git") else bits[1]
    return f"{bits[0]}/{repo}"


# ---------------------------------------------------------------------------
# Source side: what the probe request carries
# ---------------------------------------------------------------------------

_BRIDGE_USE = re.compile(
    r"\"tool_name\"\s*:\s*\"bridge\"|\"custom_type\"\s*:\s*\"team_bridge_message\""
)


def session_uses_team_bridge(root: Path, session_id: str) -> bool:
    """Whether the moving conversation evidently relies on Team Bridge.

    THERE IS NO SINGLE "THIS SESSION USES THE BRIDGE" FLAG in lop, so this reads the
    two durable traces that exist: the session is the bridge's designated
    chief-of-staff session in this device's store, or its transcript holds a
    ``bridge`` tool row or a delivered ``team_bridge_message`` card. Only then does
    the destination check its own enrolment; a session with neither gets no bridge
    warning at all (stated in the design note, §11.1). Never raises.
    """
    try:
        from local_operator.team_bridge.store import Store

        store = Store(root)
        owner = store.active_owner_readonly()
        if owner is not None and store.designated_session_for_readonly(owner) == session_id:
            return True
    except Exception:  # noqa: BLE001 — absent/damaged store reads as "not designated"
        pass
    try:
        from local_operator.network import workset

        for text in workset._transcript_texts(root / "sessions" / session_id):  # noqa: SLF001
            if _BRIDGE_USE.search(text):
                return True
    except Exception:  # noqa: BLE001 — unreadable transcript reads as "no trace"
        pass
    return False


def probe_request(
    block: Mapping[str, Any], *, session_id: str, team_bridge: bool
) -> dict[str, Any]:
    """The ``workset_probe`` frame for ``block``: one row per distinct REMOTE URL with
    the marker file NAMES of the worktrees that use it.

    MINIMUM DISCLOSURE BEFORE CONSENT (security S6, round 1). The probe runs during
    ``--dry-run`` and before the user has decided to trust the destination, so it
    carries only what the destination needs to answer: the remote (it must dial it)
    and the toolchain markers (it checks its own PATH). No worktree path, no branch,
    no head — ``ls-remote <url> HEAD`` and the throwaway push need neither.
    Mention-only worktrees are not in ``block["worktrees"]`` at all
    (``workset.mention_only``), so they are never sent.
    """
    by_url: dict[str, list[str]] = {}
    for row in block.get("worktrees") or []:
        if not isinstance(row, Mapping):
            continue
        markers = by_url.setdefault(str(row.get("remote_url") or ""), [])
        for marker in row.get("markers") or []:
            if isinstance(marker, str) and marker in TOOLCHAIN_MARKERS and marker not in markers:
                markers.append(marker)
    rows = [{"remote_url": url, "markers": sorted(markers)} for url, markers in by_url.items()]
    return {
        "op": "net_session_move",
        "phase": "workset_probe",
        "session_id": session_id,
        "probe_version": PROBE_VERSION,
        "worktrees": rows[:MAX_PROBE_ROWS],
        "team_bridge": bool(team_bridge),
    }


# ---------------------------------------------------------------------------
# Destination side: the probe itself
# ---------------------------------------------------------------------------


#: The hardened ssh the probe uses when the destination's user has not configured
#: their own (security S7): never prompt or pop an askpass dialog, never forward the
#: agent or ports to a host a remote peer chose, and never write a new host key —
#: ``StrictHostKeyChecking=yes`` checks against the destination user's own
#: known_hosts (ssh's default file) and refuses an unknown host instead of learning
#: it.
_HARDENED_SSH = (
    "ssh -o BatchMode=yes -o ForwardAgent=no -o ClearAllForwardings=yes "
    "-o StrictHostKeyChecking=yes"
)


def _user_ssh_configured(env: Mapping[str, str]) -> bool:
    if env.get("GIT_SSH_COMMAND") or env.get("GIT_SSH"):
        return True
    try:
        done = subprocess.run(
            ["git", "config", "--get", "core.sshCommand"],
            env=dict(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and bool(done.stdout.strip())


def _probe_env(extra: Mapping[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(name, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    # ssh's own askpass (a GUI dialog on a desktop with DISPLAY set) is never used for
    # a probe a remote peer triggered, whoever configured ssh.
    env["SSH_ASKPASS_REQUIRE"] = "never"
    if not _user_ssh_configured(env):
        env["GIT_SSH_COMMAND"] = _HARDENED_SSH
    env.update(extra)
    return env


#: Config every probe git call carries (security S1/S7/S8, round 1):
#: * ``protocol.allow=never`` + https/ssh ``always``: git itself enforces the URL
#:   allowlist, even after the destination's own ``insteadOf`` rewrites a URL;
#: * ``http.followRedirects=false``: a redirect to an internal service was followed
#:   and its error body echoed back (S1 repro 1); with it off, no request reached it;
#: * ``core.hooksPath=/dev/null``: the throwaway push must not run the destination's
#:   own pre-push / reference-transaction hooks with a peer-chosen URL (S8).
_BASE_CONFIG = (
    "protocol.ext.allow=never",
    "protocol.allow=never",
    "protocol.https.allow=always",
    "protocol.ssh.allow=always",
    "http.followRedirects=false",
    f"core.hooksPath={os.devnull}",
)


def _classify(stderr: str) -> str:
    """git's stderr -> one :data:`DETAIL_CODES` value. Runs on the destination only."""
    text = stderr.lower()
    if "host key verification failed" in text or "remote host identification" in text:
        return "host_key"
    if (
        "denied" in text  # "Permission denied", GitHub's "Permission to x denied to y"
        or "could not read username" in text
        or "authentication failed" in text
        or "403" in text
        or "401" in text
        or "write access" in text
    ):
        return "auth_denied"
    if "not found" in text or "does not appear to be a git repository" in text or "404" in text:
        return "not_found"
    if "ssl" in text or "certificate" in text or "tls" in text:
        return "tls"
    if (
        "could not resolve" in text
        or "failed to connect" in text
        or "connection refused" in text
        or "connection closed" in text
        or "connection timed out" in text
        or "no route to host" in text
        or "network is unreachable" in text
        or "redirect" in text
    ):
        return "unreachable"
    return "error"


def _git(args: list[str], *, env: Mapping[str, str], cwd: str | None = None) -> tuple[bool, str]:
    """One probe git call: ``(ok, detail_code)``. The code is from :data:`DETAIL_CODES`.

    THE WHOLE PROCESS GROUP DIES ON TIMEOUT (review R7, round 1): ``subprocess.run``
    killed only ``git`` and left the ``ssh`` / ``git-remote-https`` child it spawned
    running in the new session. The group exists already (``start_new_session``, which
    is also what removes the controlling tty so ssh cannot prompt), so the timeout
    kills it by its own id.
    """
    command = ["git"]
    for pair in (*_BASE_CONFIG, *_EXTRA_CONFIG):
        command += ["-c", pair]
    command += args
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # no controlling tty: ssh cannot prompt
        )
    except FileNotFoundError:
        return False, "git_missing"
    except OSError:
        return False, "error"
    try:
        _out, err = process.communicate(timeout=GIT_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        return False, "timeout"
    if process.returncode == 0:
        return True, ""
    return False, _classify((err or b"").decode("utf-8", "replace"))


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.communicate(timeout=5)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Host policy (security S1.3)
# ---------------------------------------------------------------------------


def _allowed_hosts() -> set[str]:
    named = {h.strip().lower() for h in os.environ.get(_ALLOWED_HOSTS_ENV, "").split(",")}
    return {h for h in (*named, *(h.lower() for h in _ALLOWED_HOSTS)) if h}


def _url_host_port(url: str) -> tuple[str, int]:
    if "://" in url:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        return (parts.hostname or "").lower(), parts.port or (443 if scheme == "https" else 22)
    match = _SCP.match(url)
    return ((match.group("host") if match else "").lower(), 22)


def _public(address: str) -> bool:
    """Whether ``address`` is a globally routable unicast address.

    ``is_global`` is False for loopback, RFC 1918, link-local (169.254.169.254, the
    cloud metadata address), CGNAT 100.64/10, ULA fc00::/7, unspecified and the
    documentation ranges; an IPv4-mapped IPv6 address is judged as its IPv4 form so
    ``::ffff:127.0.0.1`` is not a way round it.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global) and not ip.is_multicast


def host_policy(url: str) -> tuple[bool, list[str]]:
    """``(allowed, pin_config)`` for probing ``url`` from this device.

    REFUSED: a host that resolves to ANY non-public address, or does not resolve.
    Resolving (rather than pattern-matching the spelling) is what catches every IP
    form at once — ``0x7f000001``, ``2130706433``, ``127.1``, ``localhost`` and a DNS
    name pointing inside — because the resolver is the same one git will use. A host
    the destination's operator named (``LOCAL_OPERATOR_MOVE_PROBE_HOSTS``) is exempt:
    that is their own self-hosted git server, not a peer's choice.

    PINNED (HTTPS): the vetted addresses ride ``http.curloptResolve`` so git connects
    to what was checked, not to a second DNS answer (rebinding). ssh has no such
    knob; its window is the gap between this lookup and ssh's own, with redirects,
    agent forwarding and new host keys all refused (S7).
    """
    host, port = _url_host_port(url)
    if not host:
        return False, []
    if host in _allowed_hosts():
        return True, []
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return False, []
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses or not all(_public(address) for address in addresses):
        return False, []
    if not url.lower().startswith("https://"):
        return True, []
    pinned = ",".join(f"[{a}]" if ":" in a else a for a in addresses)
    return True, ["-c", f"http.curloptResolve={host}:{port}:{pinned}"]


def _can_fetch(url: str, env: Mapping[str, str]) -> tuple[bool, str]:
    allowed, pin = host_policy(url)
    if not allowed:
        return False, "host_refused"
    return _git([*pin, "ls-remote", "--", url, "HEAD"], env=env)


def _can_push(url: str, env: Mapping[str, str], scratch: Path) -> tuple[bool, str]:
    """``push --dry-run`` of a throwaway commit to a fresh branch name: authenticates at
    receive-pack and sends nothing (see the module docstring). The commit is of git's
    built-in empty tree, in a bare repository made for this one call."""
    allowed, pin = host_policy(url)
    if not allowed:
        return False, "host_refused"
    repo = Path(tempfile.mkdtemp(prefix="probe-", dir=scratch))
    # ``--template=``: no ``init.templateDir`` hooks are copied in (S8).
    ok, detail = _git(["init", "-q", "--bare", "--template=", str(repo)], env=env)
    if not ok:
        return False, detail
    commit_env = {
        **env,
        "GIT_AUTHOR_NAME": "lop",
        "GIT_AUTHOR_EMAIL": "lop@localhost",
        "GIT_COMMITTER_NAME": "lop",
        "GIT_COMMITTER_EMAIL": "lop@localhost",
    }
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), "commit-tree", _EMPTY_TREE, "-m", "lop move probe"],
            env=commit_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=GIT_PROBE_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False, "probe_error"
    sha = (done.stdout or b"").decode("ascii", "replace").strip()
    if done.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        return False, "probe_error"
    probe_ref = (
        "refs/heads/lop-move-probe-"
        + hashlib.sha256(f"{url}{time.time_ns()}".encode()).hexdigest()[:12]
    )
    return _git(
        [
            *pin,
            "-C",
            str(repo),
            "push",
            "--dry-run",
            "--no-verify",
            "--porcelain",
            "--",
            url,
            f"{sha}:{probe_ref}",
        ],
        env=env,
    )


#: The shape :func:`local_operator.network.identity.device_id_for` mints. The remedy
#: names the destination by this id and nothing else (security S10, round 2).
_DEVICE_ID = re.compile(r"d_[0-9a-f]{32}")


def _remedy(url: str, *, device_id: str, borrowing_github: bool) -> dict[str, str] | None:
    """The exact command that grants this destination access, when lop has one.

    Only ``github.com`` remotes have a mesh remedy (the lent ``github`` credential,
    guide ``network`` "GitHub push and PR-write"). With a share already in place the
    repository is opted out or not lent -> ``include``; with none -> ``share``. Both
    run on the LENDING device (the one whose GitHub login is shared), never here.

    THE COMMAND NAMES THE DESTINATION BY ITS DEVICE ID, NEVER ITS NAME (security S10,
    round 2). A member name is the peer's own choice and ``validate_member_name``
    admits shell metacharacters, so ``--with {name}`` let a hostile destination named
    ``node2 $(curl …|sh)`` write the command a user is told to paste on the device
    holding their GitHub login. The id is minted from a public key, is accepted by
    ``--with``, and is checked against its exact shape here; anything else gets no
    command at all. Every token is ``shlex.quote``d as well, so a future input
    cannot reopen this. The surrounding sentence still shows the readable name.
    """
    repository = github_repository(url) or github_repository(https_rewrite(url))
    if not repository:
        return None
    if borrowing_github:
        command = f"lop network credential github include {shlex.quote(repository)}"
    else:
        if not _DEVICE_ID.fullmatch(device_id):
            return None
        command = f"lop network credential share github --with {shlex.quote(device_id)}"
    return {"run_on": "lending_device", "command": command}


def _check_one(
    url: str, *, env: Mapping[str, str], scratch: Path, have_git: bool
) -> dict[str, Any]:
    """The destination's answer for one remote: a ``clone_check``, a detail CODE and,
    for an SSH remote it cannot use, whether the HTTPS spelling fetches/pushes.

    No remedy and no rewritten URL are returned (security S3, round 1): the asker
    rebuilds both from the URL it sent and these booleans, so a hostile destination
    cannot put text of its choosing on the consent screen.
    """
    if not url:
        return {"clone_check": "no_remote"}
    if classify_url(url) is None:
        return {"clone_check": "destination_cannot_clone", "detail": "url_refused"}
    if not have_git:
        return {"clone_check": "destination_cannot_clone", "detail": "git_missing"}
    fetch_ok, fetch_detail = _can_fetch(url, env)
    if fetch_ok:
        push_ok, push_detail = _can_push(url, env, scratch)
        if push_ok:
            return {"clone_check": "ok"}
        return {"clone_check": "destination_cannot_push", "detail": push_detail}
    answer: dict[str, Any] = {"clone_check": "destination_cannot_fetch", "detail": fetch_detail}
    rewrite = https_rewrite(url)
    if rewrite:
        rewrite_ok, _ = _can_fetch(rewrite, env)
        if rewrite_ok:
            push_ok, _ = _can_push(rewrite, env, scratch)
            answer["https_rewrite"] = {"fetch": True, "push": bool(push_ok)}
    return answer


def missing_tools(markers: Iterable[str]) -> list[str]:
    """Toolchains ``markers`` need that this device's PATH lacks, sorted, unique."""
    needed: list[str] = []
    for marker in markers:
        for tool in TOOLCHAIN_MARKERS.get(str(marker), ()):
            if tool not in needed:
                needed.append(tool)
    return sorted(tool for tool in needed if shutil.which(tool) is None)


def team_bridge_enrolled(root: Path) -> bool:
    """Whether THIS device has Team Bridge on and a committed enrolment. Never raises."""
    try:
        from local_operator.config import read_config_values
        from local_operator.team_bridge.settings import TeamBridgeSettings
        from local_operator.team_bridge.store import Store

        values = read_config_values(Path(root))
        if values is None or not TeamBridgeSettings.from_values(values).enabled:
            return False
        return Store(Path(root)).active_owner_readonly() is not None
    except Exception:  # noqa: BLE001 — unreadable reads as "not enrolled" (a warning)
        return False


def run_probe(root: Path, frame: Mapping[str, Any], *, device: str = "") -> dict[str, Any]:
    """Serve ``workset_probe`` on the DESTINATION. Read-only, bounded, never raises
    for a bad row (each row answers on its own). ``device`` is unused since the asker
    builds the remedy itself (S3); kept so older call sites keep their shape."""
    del device
    rows = [row for row in frame.get("worktrees") or [] if isinstance(row, Mapping)]
    rows = rows[:MAX_PROBE_ROWS]
    urls = list(dict.fromkeys(str(row.get("remote_url") or "") for row in rows))
    lent: dict[str, str] = {}
    if any(
        github_repository(url) and (url.startswith("https://") or https_rewrite(url))
        for url in urls
    ):
        try:
            from local_operator.network.credentials import github as github_credentials

            lent, _token = github_credentials.borrowed_git_env(root=Path(root))
        except Exception:  # noqa: BLE001 — no lent credential: probe with our own
            lent = {}
    env = _probe_env(lent)
    have_git = shutil.which("git") is not None
    checks: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="lop-move-probe-") as scratch_text:
        scratch = Path(scratch_text)
        # One check per distinct URL: the push probe uses its own throwaway branch
        # name, so two worktrees of one repository get one answer.
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                url: pool.submit(_check_one, url, env=env, scratch=scratch, have_git=have_git)
                for url in urls
            }
            for url, future in futures.items():
                try:
                    checks[url] = future.result()
                except Exception:  # noqa: BLE001 — one row's bug must not sink the rest
                    logger.warning("workset probe: a row failed", exc_info=True)
                    checks[url] = {
                        "clone_check": "destination_cannot_clone",
                        "detail": "probe_error",
                    }
    answered = []
    for row in rows:
        url = str(row.get("remote_url") or "")
        item = {"remote_url": url, **checks.get(url, {})}
        item["missing_tools"] = missing_tools(row.get("markers") or [])
        answered.append(item)
    from local_operator.network.workset import WORKSET_TRANSFER_VERSION

    answer: dict[str, Any] = {
        "ok": True,
        "probe_version": PROBE_VERSION,
        # Whether THIS device can receive the files (PR B): the offload preflight
        # recomputes its manifest as "they stay" for a destination below the
        # transfer protocol, so the user is never shown a send that will not happen.
        "working_set_version": WORKSET_TRANSFER_VERSION,
        "worktrees": answered,
        # Whether a remedy should be ``include`` (a share exists) or ``share``; the
        # asker builds the command (S3).
        "borrowing_github": bool(lent),
    }
    if frame.get("team_bridge"):
        answer["team_bridge"] = {"needed": True, "enrolled": team_bridge_enrolled(Path(root))}
    return answer


# ---------------------------------------------------------------------------
# Destination admission: owner binding, rate limit, audit summary (security S1)
# ---------------------------------------------------------------------------

#: Probes one asking device may trigger per window. NOT MEASURED against a traffic
#: distribution (there is none yet): a move attempt asks at most twice (the CLI's
#: prompt re-probes once, R6) and the TUI once, so 10 per minute is several times any
#: interactive use while bounding a peer that loops the phase to scan hosts.
PROBE_RATE_LIMIT = 10
PROBE_RATE_WINDOW_S = 60.0
_rate_lock = threading.Lock()
_rate_seen: dict[str, list[float]] = {}


def rate_limited(device_id: str, *, now: float | None = None) -> bool:
    """Record one probe by ``device_id``; True when it exceeds the window's budget."""
    moment = time.monotonic() if now is None else now
    with _rate_lock:
        recent = [t for t in _rate_seen.get(device_id, []) if moment - t < PROBE_RATE_WINDOW_S]
        if len(recent) >= PROBE_RATE_LIMIT:
            _rate_seen[device_id] = recent
            return True
        recent.append(moment)
        _rate_seen[device_id] = recent
        return False


def audit_summary(frame: Mapping[str, Any], answer: Mapping[str, Any] | None) -> dict[str, Any]:
    """Bounded audit metadata for one served probe: host names and result codes only
    (never a path, a URL's path or userinfo)."""
    hosts: list[str] = []
    for row in frame.get("worktrees") or []:
        if isinstance(row, Mapping):
            host = _url_host_port(str(row.get("remote_url") or ""))[0]
            if host and host not in hosts:
                hosts.append(host)
    results: dict[str, int] = {}
    for item in (answer or {}).get("worktrees") or []:
        if isinstance(item, Mapping):
            key = str(item.get("detail") or item.get("clone_check") or "")
            results[key] = results.get(key, 0) + 1
    return {
        "hosts": ",".join(hosts[:16]),
        "results": ",".join(f"{k}={v}" for k, v in sorted(results.items())),
    }


# ---------------------------------------------------------------------------
# Folding the answer into the preflight block
# ---------------------------------------------------------------------------


#: The per-row fields ``apply_probe`` writes, and the block field. THIS DEVICE OWNS
#: THEM (security S11, round 3): on a recall the block is the REMOTE owner's manifest,
#: and a field the probe only ever ADDED let a hostile owner pre-fill ``remedy`` with
#: any command and have the consent card tell the user to paste it.
PROBE_ROW_FIELDS = ("remedy", "https_rewrite", "clone_check", "clone_detail", "missing_tools")
PROBE_BLOCK_FIELDS = ("destination_probe",)


#: Fields only the restoring/probing device writes (see ``scrub_peer_block``).
PEER_RESULT_ROW_FIELDS = ("recipes", "dest_path", "remapped", "branch_moved", "mode")
PEER_RESULT_BLOCK_FIELDS = ("notes", "import")


#: The only shape ``workset.scope_digest`` mints.
_SCOPE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def strip_probe_fields(block: dict[str, Any]) -> dict[str, Any]:
    """Drop every probe-owned field from ``block`` in place (and return it).

    ``clone_check`` is put back to the unprobed default every manifest row carries
    (``workset.WorktreeFacts.to_dict``), so a row the answer does not name still reads
    ``pending_destination`` rather than losing the field."""
    for row in block.get("worktrees") or []:
        if isinstance(row, dict):
            for name in PROBE_ROW_FIELDS:
                row.pop(name, None)
            row["clone_check"] = "pending_destination"
    for name in PROBE_BLOCK_FIELDS:
        block.pop(name, None)
    return block


def scrub_peer_block(block: dict[str, Any]) -> dict[str, Any]:
    """Make a working-set block a PEER sent safe to render here, in place.

    Called where a block crosses the link into this device (the recall preflight,
    a peer's refusal, the owner's arrival block). The audit behind it (security S11,
    round 3): the only fields any surface renders as an instruction or a command are
    the probe's ``remedy`` / ``https_rewrite`` (rebuilt here by ``apply_probe``),
    ``left_behind[].recipe.run`` ("recreate with: <run>" on the consent card, the
    receipt and the model's arrival note) and ``scope_digest``, which the CLI prints
    inside "re-run with --confirm-working-set <digest>". The first are stripped; a
    recipe survives only when it is one this build writes (``workset.trusted_recipe``);
    a digest only in the ``sha256:<64 hex>`` shape ``workset.scope_digest`` mints —
    dropping it fails closed, since the owner refuses a move confirming "". Everything
    else is shown as data — paths, branch and device names pass through ``clean`` on
    every text surface, and enum fields (``state``, ``reason``, ``why``, ``consent``)
    render through closed tables.
    """
    from local_operator.network import workset

    strip_probe_fields(block)
    # THE TRANSFER HALF'S RESULT FIELDS ARE ALSO THIS DEVICE'S (PR B, #4245): a row's
    # ``recipes`` ("To recreate its dependencies, run in <cwd>: <run>"), ``dest_path``,
    # ``remapped`` and ``branch_moved``, and the block's ``notes`` and ``import``
    # (whose ``remedy`` is a command) are written only by the device that restored
    # or probed — ``workset_move.arrival_receipt`` merges a restore row's fields only
    # when the restore set them, so a peer's pre-filled value would otherwise survive
    # onto the receipt. ``transferred: true`` is likewise only ever set locally (no
    # block that crosses the link before a restore legitimately says so). The source's
    # offload receipt reads the destination's answer through its own closed reducer
    # (``workset_move.source_receipt``), not this block.
    for row in block.get("worktrees") or []:
        if isinstance(row, dict):
            for name in PEER_RESULT_ROW_FIELDS:
                row.pop(name, None)
    for name in PEER_RESULT_BLOCK_FIELDS:
        block.pop(name, None)
    if block.get("transferred") is True:
        block["transferred"] = False
    if "scope_digest" in block and not _SCOPE_DIGEST.fullmatch(str(block["scope_digest"])):
        block.pop("scope_digest")
    for item in block.get("left_behind") or []:
        if isinstance(item, dict) and "recipe" in item:
            recipe = workset.trusted_recipe(item.get("recipe"))
            if recipe is None:
                item.pop("recipe", None)
            else:
                item["recipe"] = recipe
    return block


def apply_probe(
    block: dict[str, Any], answer: Mapping[str, Any] | None, *, device_id: str = ""
) -> dict[str, Any]:
    """Fold the destination's answer into ``block`` in place (and return it).

    ``answer=None`` means the destination does not serve the probe (an older lop):
    every row reads ``unknown_destination``, which a surface renders as "could not
    check", never as a failure. Rows are matched by ``remote_url``; a row the answer
    does not name keeps ``pending_destination``. No row's ``state`` changes — see the
    module docstring for why the probe flags rather than leaves behind.

    THE DESTINATION SUPPLIES NO TEXT (security S3, round 1). Only a ``clone_check``
    from :data:`CLONE_CHECKS`, a detail from :data:`DETAIL_CODES`, the ``fetch`` /
    ``push`` / ``borrowing_github`` booleans and tool names that are values of
    :data:`TOOLCHAIN_MARKERS` are kept. The HTTPS spelling and the remedy command are
    rebuilt HERE from the URL this device sent and ``device_id`` (the destination's
    mesh id — never its self-chosen name, security S10), so a hostile destination
    cannot write a command, markup or a terminal escape onto the consent screen.

    NOR DOES THE OWNER (security S11, round 3): on a recall ``block`` came from the
    remote owner, so every probe-owned field is dropped FIRST — before the
    ``answer is None`` return too — and only what is rebuilt below survives.
    """
    strip_probe_fields(block)
    rows = [row for row in block.get("worktrees") or [] if isinstance(row, dict)]
    if answer is None:
        for row in rows:
            row["clone_check"] = "unknown_destination"
        block["destination_probe"] = {"state": "unknown_destination"}
        return block
    by_url = {
        str(item.get("remote_url") or ""): item
        for item in answer.get("worktrees") or []
        if isinstance(item, Mapping)
    }
    known_tools = {tool for tools in TOOLCHAIN_MARKERS.values() for tool in tools}
    borrowing = answer.get("borrowing_github") is True
    missing_union: list[str] = []
    for row in rows:
        url = str(row.get("remote_url") or "")
        item = by_url.get(url)
        if item is None:
            continue
        check = str(item.get("clone_check") or "")
        check = check if check in CLONE_CHECKS else "unknown_destination"
        row["clone_check"] = check
        detail = str(item.get("detail") or "")
        if detail:
            row["clone_detail"] = detail if detail in DETAIL_CODES else "error"
        remedy_url = url
        rewrite = item.get("https_rewrite")
        rewritten = https_rewrite(url)
        if isinstance(rewrite, Mapping) and rewrite.get("fetch") is True and rewritten:
            push = rewrite.get("push") is True
            row["https_rewrite"] = {"url": rewritten, "fetch": True, "push": push}
            remedy_url = rewritten
            needs_remedy = not push
        else:
            # QA Q6: a policy refusal has no credential remedy to offer.
            needs_remedy = check in (
                "destination_cannot_fetch",
                "destination_cannot_push",
            ) and row.get("clone_detail") not in ("host_refused", "url_refused")
        if needs_remedy:
            remedy = _remedy(remedy_url, device_id=device_id, borrowing_github=borrowing)
            if remedy:
                row["remedy"] = remedy
        tools = sorted(
            {str(t) for t in item.get("missing_tools") or [] if isinstance(t, str)} & known_tools
        )
        row["missing_tools"] = tools
        missing_union.extend(t for t in tools if t not in missing_union)
    probe: dict[str, Any] = {"state": "answered", "missing_tools": sorted(missing_union)}
    if "working_set_version" in answer:
        # Absent from a destination that predates the transfer: the reader treats a
        # missing key as 0 ("cannot receive files"), so it is not invented here.
        try:
            probe["working_set_version"] = int(answer.get("working_set_version") or 0)
        except (TypeError, ValueError):
            probe["working_set_version"] = 0
    # Whether the destination accepts this device's files (``workset_import`` on its
    # row for us, security-lead ruling #4245). Only a real boolean is kept; absent or
    # malformed reads as "not granted" at the reader, which then plans no transfer.
    if isinstance(answer.get("import_granted"), bool):
        probe["import_granted"] = answer["import_granted"]
    bridge = answer.get("team_bridge")
    if isinstance(bridge, Mapping) and bridge.get("needed"):
        probe["team_bridge"] = {"needed": True, "enrolled": bridge.get("enrolled") is True}
    block["destination_probe"] = probe
    return block


# ---------------------------------------------------------------------------
# Arrival: the one continuation note (model-facing, English by design)
# ---------------------------------------------------------------------------


def arrival_note(
    block: Mapping[str, Any] | None,
    *,
    session_id: str,
    source: str,
    destination: str,
    setup_pending: Sequence[tuple[str, str]] = (),
) -> str:
    """The system note a moved ACTIVE conversation receives once on arrival.

    MODEL-FACING, so English by design (AGENTS.md i18n exception): it tells the
    model which paths it worked in, that they did NOT come (PR A carries no files),
    where they are and how to get them, what was left behind, and which toolchains
    this device lacks (recomputed HERE from the block's marker names).

    ``setup_pending`` is ``(cwd, run)`` per dependency recipe THIS device's own setup
    plan (``workset_setup.plan_setup``) still has to run. The note says so and names
    ONLY the non-executing ``lop sessions setup-deps <id> --status``. The receiver
    boundary (security ruling on #4378): moving the files, the mesh authority and
    the agent's shell are not consent to run a package manager's lifecycle scripts
    here, so the note never tells the agent to run setup and never offers ``--yes``;
    running it stays a decision the operator makes on this device.
    """
    lines = [
        f"[session move] This conversation was just moved from {source} to {destination} "
        "while it was active. Before continuing, note what did and did not come with it."
    ]
    rows = [row for row in (block or {}).get("worktrees") or [] if isinstance(row, Mapping)]
    if rows:
        lines.append("Working files (git worktrees) this conversation used:")
        for row in rows:
            path = str(row.get("path") or "")
            branch = str(row.get("branch") or "HEAD")
            state = str(row.get("state") or "")
            if state in ("restored", "carried"):
                # QA Q1: name where the files ARE. A beside or remapped restore leaves
                # ``path`` holding something else here (the user's own files), and an
                # agent following the note would edit the wrong tree.
                dest = str(row.get("dest_path") or "")
                if dest and dest != path:
                    lines.append(
                        f"- {path} (branch {branch}): restored on this device at {dest}; "
                        f"work there, not in {path} (on this device that path holds "
                        "other files or is not where the worktree landed)."
                    )
                else:
                    lines.append(f"- {path} (branch {branch}): restored on this device.")
                continue
            lines.append(
                f"- {path} (branch {branch}): NOT on this device; it stayed on {source}. "
                f"To work on it here, push branch {branch} from {source} and fetch it here, "
                f"or move this conversation back with `lop sessions move {session_id} "
                f"--to {source}`."
            )
            tools = missing_tools(row.get("markers") or [])
            if tools:
                lines.append(f"  This device lacks: {', '.join(tools)}.")
    left = [item for item in (block or {}).get("left_behind") or [] if isinstance(item, Mapping)]
    for item in left[:20]:
        recipe = item.get("recipe") if isinstance(item.get("recipe"), Mapping) else {}
        run = str((recipe or {}).get("run") or "")
        suffix = f" (recreate with: {run})" if run else ""
        lines.append(f"- Left behind: {item.get('path')}{suffix}")
    excluded = [item for item in (block or {}).get("excluded") or [] if isinstance(item, Mapping)]
    for item in excluded[:20]:
        lines.append(f"- Withheld (secret-shaped): {item.get('path')}")
    pending = [(str(cwd), str(run)) for cwd, run in setup_pending][:20]
    if pending:
        lines.append(
            "Dependency trees (node_modules, .venv) did not move and are not set up on "
            "this device yet:"
        )
        lines.extend(f"- {cwd}: {run}" for cwd, run in pending)
        lines.append(
            "Creating them runs a package manager, which can execute the repository's "
            "install scripts, so it needs the user's own decision on this device; moving "
            "the conversation did not approve it. Do not run the setup or these commands "
            "yourself: tell the user it is pending. To show the plan and its status "
            f"without running anything: lop sessions setup-deps {session_id} --status"
        )
    if len(lines) == 1:
        lines.append("No working files were recorded for this conversation.")
    lines.append(
        "Tell the user about anything missing before relying on these paths; do not "
        "assume the files above exist here."
    )
    return "\n".join(lines)


__all__ = [
    "ASK_TIMEOUT_S",
    "CLONE_CHECKS",
    "DETAIL_CODES",
    "PROBE_RATE_LIMIT",
    "PROBE_VERSION",
    "TOOLCHAIN_MARKERS",
    "apply_probe",
    "arrival_note",
    "audit_summary",
    "classify_url",
    "github_repository",
    "host_policy",
    "https_rewrite",
    "missing_tools",
    "probe_request",
    "rate_limited",
    "run_probe",
    "scrub_peer_block",
    "session_uses_team_bridge",
    "strip_probe_fields",
    "team_bridge_enrolled",
]

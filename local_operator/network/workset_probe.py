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
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

PROBE_VERSION = 1

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

_HOST = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,252}[A-Za-z0-9])?$")
_SCP = re.compile(
    r"^(?:(?P<user>[A-Za-z0-9._-]{1,64})@)?(?P<host>[A-Za-z0-9.-]{2,253}):(?P<path>[^:]+)$"
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
    """The ``workset_probe`` frame for ``block``: per row only path, remote, branch and
    marker NAMES — nothing the destination could not already learn from the manifest."""
    rows = []
    for row in block.get("worktrees") or []:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "path": str(row.get("path") or ""),
                "remote_url": str(row.get("remote_url") or ""),
                "branch": str(row.get("branch") or ""),
                "markers": [str(m) for m in row.get("markers") or [] if isinstance(m, str)],
            }
        )
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


def _probe_env(extra: Mapping[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(name, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.update(extra)
    return env


def _git(args: list[str], *, env: Mapping[str, str], cwd: str | None = None) -> tuple[bool, str]:
    command = ["git", "-c", "protocol.ext.allow=never"]
    for pair in _EXTRA_CONFIG:
        command += ["-c", pair]
    command += args
    try:
        done = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=GIT_PROBE_TIMEOUT_S,
            start_new_session=True,  # no controlling tty: ssh cannot prompt
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except OSError as exc:
        return False, type(exc).__name__
    # The FIRST line is the cause (``ERROR: Permission ... denied``, ``ssh: connect
    # ... refused``, ``fatal: could not read Username``); git's trailing lines are a
    # generic "make sure you have the correct access rights" paragraph.
    lines = [
        line
        for line in (done.stderr or b"").decode("utf-8", "replace").splitlines()
        if line.strip()
    ]
    return done.returncode == 0, (lines[0].strip()[:200] if lines else "")


def _can_fetch(url: str, env: Mapping[str, str]) -> tuple[bool, str]:
    return _git(["ls-remote", "--", url, "HEAD"], env=env)


def _can_push(url: str, env: Mapping[str, str], scratch: Path) -> tuple[bool, str]:
    """``push --dry-run`` of a throwaway commit to a fresh branch name: authenticates at
    receive-pack and sends nothing (see the module docstring). The commit is of git's
    built-in empty tree, in a bare repository made for this one call."""
    repo = Path(tempfile.mkdtemp(prefix="probe-", dir=scratch))
    ok, detail = _git(["init", "-q", "--bare", str(repo)], env=env)
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
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, type(exc).__name__
    sha = (done.stdout or b"").decode("ascii", "replace").strip()
    if done.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        return False, "probe_commit"
    probe_ref = (
        "refs/heads/lop-move-probe-"
        + hashlib.sha256(f"{url}{time.time_ns()}".encode()).hexdigest()[:12]
    )
    return _git(
        ["-C", str(repo), "push", "--dry-run", "--porcelain", "--", url, f"{sha}:{probe_ref}"],
        env=env,
    )


def _remedy(url: str, *, device: str, borrowing_github: bool) -> dict[str, str] | None:
    """The exact command that grants this destination access, when lop has one.

    Only ``github.com`` remotes have a mesh remedy (the lent ``github`` credential,
    guide ``network`` "GitHub push and PR-write"). With a share already in place the
    repository is opted out or not lent -> ``include``; with none -> ``share``. Both
    run on the LENDING device (the one whose GitHub login is shared), never here.
    """
    repository = github_repository(url) or github_repository(https_rewrite(url))
    if not repository:
        return None
    if borrowing_github:
        command = f"lop network credential github include {repository}"
    else:
        command = f"lop network credential share github --with {device}"
    return {"run_on": "lending_device", "command": command}


def _check_one(
    url: str,
    *,
    env: Mapping[str, str],
    scratch: Path,
    device: str,
    borrowing_github: bool,
    have_git: bool,
) -> dict[str, Any]:
    if not url:
        return {"clone_check": "no_remote"}
    if classify_url(url) is None:
        return {"clone_check": "destination_cannot_clone", "detail": "url_refused"}
    if not have_git:
        return {"clone_check": "destination_cannot_clone", "detail": "git_missing"}
    answer: dict[str, Any] = {}
    fetch_ok, fetch_detail = _can_fetch(url, env)
    if not fetch_ok:
        answer = {"clone_check": "destination_cannot_fetch", "detail": fetch_detail}
        rewrite = https_rewrite(url)
        if rewrite:
            rewrite_ok, _ = _can_fetch(rewrite, env)
            if rewrite_ok:
                push_ok, _ = _can_push(rewrite, env, scratch)
                answer["https_rewrite"] = {"url": rewrite, "fetch": True, "push": push_ok}
                if not push_ok:
                    remedy = _remedy(rewrite, device=device, borrowing_github=borrowing_github)
                    if remedy:
                        answer["remedy"] = remedy
                return answer
    else:
        push_ok, push_detail = _can_push(url, env, scratch)
        if push_ok:
            return {"clone_check": "ok"}
        answer = {"clone_check": "destination_cannot_push", "detail": push_detail}
    remedy = _remedy(url, device=device, borrowing_github=borrowing_github)
    if remedy:
        answer["remedy"] = remedy
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


def run_probe(root: Path, frame: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    """Serve ``workset_probe`` on the DESTINATION. Read-only, bounded, never raises
    for a bad row (each row answers on its own); ``device`` is this device's name,
    used only in a ``share --with`` remedy."""
    rows = [row for row in frame.get("worktrees") or [] if isinstance(row, Mapping)]
    rows = rows[:MAX_PROBE_ROWS]
    urls = {str(row.get("remote_url") or "") for row in rows}
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
        keys = list(dict.fromkeys(str(r.get("remote_url") or "") for r in rows))
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                key: pool.submit(
                    _check_one,
                    key,
                    env=env,
                    scratch=scratch,
                    device=device,
                    borrowing_github=bool(lent),
                    have_git=have_git,
                )
                for key in keys
            }
            for key, future in futures.items():
                try:
                    checks[key] = future.result()
                except Exception:  # noqa: BLE001 — one row's bug must not sink the rest
                    logger.warning("workset probe: a row failed", exc_info=True)
                    checks[key] = {
                        "clone_check": "destination_cannot_clone",
                        "detail": "probe_error",
                    }
    answered = []
    for row in rows:
        key = str(row.get("remote_url") or "")
        item = {"path": str(row.get("path") or ""), **checks.get(key, {})}
        item["missing_tools"] = missing_tools(row.get("markers") or [])
        answered.append(item)
    answer: dict[str, Any] = {"ok": True, "probe_version": PROBE_VERSION, "worktrees": answered}
    if frame.get("team_bridge"):
        answer["team_bridge"] = {"needed": True, "enrolled": team_bridge_enrolled(Path(root))}
    return answer


# ---------------------------------------------------------------------------
# Folding the answer into the preflight block
# ---------------------------------------------------------------------------


def apply_probe(block: dict[str, Any], answer: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fold the destination's answer into ``block`` in place (and return it).

    ``answer=None`` means the destination does not serve the probe (an older lop):
    every row reads ``unknown_destination``, which a surface renders as "could not
    check", never as a failure. Rows are matched by ``path``; a row the answer does
    not name keeps ``pending_destination``. No row's ``state`` changes — see the
    module docstring for why the probe flags rather than leaves behind.
    """
    rows = [row for row in block.get("worktrees") or [] if isinstance(row, dict)]
    if answer is None:
        for row in rows:
            row["clone_check"] = "unknown_destination"
        block["destination_probe"] = {"state": "unknown_destination"}
        return block
    by_path = {
        str(item.get("path") or ""): item
        for item in answer.get("worktrees") or []
        if isinstance(item, Mapping)
    }
    missing_union: list[str] = []
    for row in rows:
        item = by_path.get(str(row.get("path") or ""))
        if item is None:
            continue
        check = str(item.get("clone_check") or "")
        row["clone_check"] = check if check in CLONE_CHECKS else "unknown_destination"
        for key in ("detail", "https_rewrite", "remedy"):
            if item.get(key):
                row[key if key != "detail" else "clone_detail"] = item[key]
        tools = [str(t) for t in item.get("missing_tools") or [] if isinstance(t, str)]
        row["missing_tools"] = tools
        missing_union.extend(t for t in tools if t not in missing_union)
    probe: dict[str, Any] = {"state": "answered", "missing_tools": sorted(missing_union)}
    bridge = answer.get("team_bridge")
    if isinstance(bridge, Mapping) and bridge.get("needed"):
        probe["team_bridge"] = {"needed": True, "enrolled": bool(bridge.get("enrolled"))}
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
) -> str:
    """The system note a moved ACTIVE conversation receives once on arrival.

    MODEL-FACING, so English by design (AGENTS.md i18n exception): it tells the
    model which paths it worked in, that they did NOT come (PR A carries no files),
    where they are and how to get them, what was left behind, and which toolchains
    this device lacks (recomputed HERE from the block's marker names).
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
    "PROBE_VERSION",
    "TOOLCHAIN_MARKERS",
    "apply_probe",
    "arrival_note",
    "classify_url",
    "github_repository",
    "https_rewrite",
    "missing_tools",
    "probe_request",
    "run_probe",
    "session_uses_team_bridge",
    "team_bridge_enrolled",
]

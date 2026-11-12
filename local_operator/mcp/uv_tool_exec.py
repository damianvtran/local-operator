"""Exec an installed uv tool directly instead of through ``uvx``.

Why this exists
---------------
The usual way to configure a Python MCP server is ``"command": "uvx",
"args": ["workspace-mcp", ...]``. ``uvx`` is ``uv tool uvx``: it resolves the
tool, then spawns the tool's console script as a CHILD and stays alive as its
parent for the server's whole lifetime — and while it lives it holds
``~/.cache/uv/.lock`` open (measured 2026-11-11 with ``lsof`` on a live
session's ``uv tool uvx workspace-mcp`` parent; the child was the installed
``~/.local/share/uv/tools/workspace-mcp/bin/workspace-mcp``). Every lop session
starts its own copy of the server, so ~36 concurrent sessions meant ~36 lock
holders: ``uv cache prune`` could never acquire the lock and a 10 GiB cache sat
unreclaimable on a disk that was critically low.

When the tool is already installed (``uv tool install workspace-mcp``) and the
config asks for nothing but the bare name, ``uvx`` would run exactly that
installed console script anyway, so launching the script ourselves gives the
same program with the same args and cwd — and no uv parent, no lock.

The env must match too: ``uvx`` hands its child the caller's env plus two
keys (measured 2026-11-11 with uv 0.10.8 by dumping a real ``uvx`` child's
``os.environ``): ``PATH`` with ``<tool_dir>/<pkg>/bin`` prepended, and
``UV=<the uv binary>``. No ``VIRTUAL_ENV`` is set. Without the PATH prepend a
server that shells out to ``python``/``pip`` or a sibling console script by
name would find the system interpreter instead of the tool's own, so
:func:`resolve_uv_tool_command` returns that overlay with the command.

Fallback rules (the original command is used UNCHANGED whenever any holds)
-------------------------------------------------------------------------
* The command is not ``uvx``, nor ``uv tool run`` / ``uv tool uvx``.
* Any uv option precedes the package (``--from``, ``--with``, ``--python``,
  ``-q``, ...): options change WHICH environment uvx builds, so the installed
  tool may not be what uvx would run.
* The package spec is not a bare, already-normalised name: a version pin
  (``==``, ``@latest``, ``>=``), extras (``pkg[x]``), a path or a URL.
* The server's own ``env`` sets a ``UV_*`` variable other than
  ``UV_TOOL_DIR`` (``UV_PYTHON`` and friends alter uvx's choice the same way
  an option would). ``UV_TOOL_DIR`` is honoured by asking uv itself, below.
* ``uv tool dir`` fails, times out, or ``<tool_dir>/<pkg>/bin/<pkg>`` is
  missing or not executable.
* The install is broken in a way ``uvx`` would repair by rebuilding the
  environment: ``<tool_dir>/<pkg>/bin/python`` dangles (its base interpreter
  was removed, e.g. a Homebrew or uv-managed Python upgrade) or
  ``uv-receipt.toml`` is missing (uvx ignores a tool dir without a receipt).
  Exec'ing the script directly would fail with exit 127 where uvx recovers.
* Windows (the console script lives under ``Scripts\\`` as an ``.exe`` shim;
  not handled here).
* The operator opted out with ``LOCAL_OPERATOR_MCP_UVX_DIRECT=0``.

The tool directory comes from ``uv tool dir`` run with the CHILD's environment
(so a ``UV_TOOL_DIR`` / ``XDG_DATA_HOME`` the server would see is respected,
including a ``uv.toml`` ``tool-dir`` setting we would otherwise have to
re-implement) and the server's cwd, bounded by a timeout and cached per
(uv binary, relevant env, cwd). Only successful answers are cached for good;
a failure or timeout is remembered for :data:`FAILED_LOOKUP_TTL_S` so one slow
lookup under load does not disable direct exec for the life of the process.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

#: Opt-out: set to ``0`` / ``false`` / ``no`` / ``off`` to always launch the
#: configured ``uvx`` command verbatim.
OPT_OUT_ENV = "LOCAL_OPERATOR_MCP_UVX_DIRECT"

#: Bound on ``uv tool dir``. Measured 2026-11-11 on the operator's machine
#: under ~36 live sessions: n=10, median ~0.06 s, max 0.24 s (uv 0.10.8). Five
#: seconds sits well above the max; on expiry we simply keep ``uvx``.
UV_TOOL_DIR_TIMEOUT_S = 5.0

#: How long a failed or timed-out ``uv tool dir`` is remembered. Long enough
#: that a burst of server spawns does not each pay up to the 5 s timeout again;
#: short enough that a transient stall under load re-enables direct exec soon.
FAILED_LOOKUP_TTL_S = 60.0

# A bare package name exactly as uv normalises it for the tool directory
# (PEP 503: lowercase, runs of ``-_.`` collapsed to ``-``). Anything else —
# capitals, underscores, pins, extras, paths, URLs — falls back, because the
# directory name and executable name would need uv's own resolution to match.
_BARE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9]))*$")

# Environment keys that decide where uv keeps tools; part of the cache key so
# a server pointing elsewhere does not reuse another server's answer.
# XDG_CONFIG_HOME locates the user ``uv.toml`` (which may set ``tool-dir``).
# ``UV_CONFIG_FILE`` is deliberately absent: a server env setting it already
# falls back via the ``UV_*`` rule, and lop's own env does not set it.
_TOOL_DIR_ENV_KEYS = ("UV_TOOL_DIR", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "HOME")

_CacheKey = tuple[str, tuple[str | None, ...], str | None]
# Value: (tool dir or None, monotonic expiry; None = never expires).
_tool_dir_cache: dict[_CacheKey, tuple[Path | None, float | None]] = {}
_tool_dir_lock = threading.Lock()


class UvToolExec(NamedTuple):
    """How to launch an installed uv tool without the ``uvx`` parent."""

    command: str
    args: list[str]
    #: Keys to overlay on the child's env so it sees what ``uvx`` would give it.
    env: dict[str, str]


def _opted_out() -> bool:
    raw = os.environ.get(OPT_OUT_ENV)
    return raw is not None and raw.strip().lower() in {"0", "false", "no", "off"}


def _uvx_package(command: str, args: list[str]) -> tuple[str, list[str]] | None:
    """Return ``(package, tool_args)`` for a bare-name uvx invocation, else ``None``."""
    name = Path(command).name
    if name == "uvx":
        rest = args
    elif name == "uv" and len(args) >= 2 and args[0] == "tool" and args[1] in ("run", "uvx"):
        rest = args[2:]
    else:
        return None
    if not rest:
        return None
    package = rest[0]
    # Any leading option (``--from``, ``--with``, ``-q``, ...) is a uv option:
    # it can change the environment uvx would run, so keep uvx.
    if package.startswith("-") or not _BARE_NAME_RE.match(package):
        return None
    return package, list(rest[1:])


def is_uvx_candidate(command: str, args: list[str]) -> bool:
    """Cheap, I/O-free check: could this command be rewritten at all?"""
    return _uvx_package(command, args) is not None


def _uv_binary(command: str, path_env: str | None) -> str | None:
    """The ``uv`` that the configured command would have used."""
    cmd_path = Path(command)
    if cmd_path.is_absolute():
        located: str | None = command
    else:
        located = shutil.which(command, path=path_env)
    if located is None:
        return None
    if Path(located).name == "uv":
        return located
    # ``uvx`` ships beside ``uv``; prefer that sibling so both agree on version.
    sibling = Path(located).with_name("uv")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("uv", path=path_env)


def _query_tool_dir(uv: str, env: Mapping[str, str], cwd: str | None) -> Path | None:
    try:
        proc = subprocess.run(
            [uv, "tool", "dir"],
            env=dict(env),
            # The server's cwd, so a project-level uv config there is seen the
            # way uvx (spawned in that cwd) would see it.
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=UV_TOOL_DIR_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("uv tool dir failed (%s); keeping uvx", type(exc).__name__)
        return None
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out:
        logger.debug("uv tool dir exited %s; keeping uvx", proc.returncode)
        return None
    return Path(out.splitlines()[-1].strip())


def uv_tool_dir(uv: str, env: Mapping[str, str], cwd: str | None = None) -> Path | None:
    """Cached, time-bounded ``uv tool dir`` under the child's environment."""
    key: _CacheKey = (uv, tuple(env.get(k) for k in _TOOL_DIR_ENV_KEYS), cwd)
    now = time.monotonic()
    with _tool_dir_lock:
        hit = _tool_dir_cache.get(key)
        if hit is not None and (hit[1] is None or hit[1] > now):
            return hit[0]
    result = _query_tool_dir(uv, env, cwd)
    expiry = None if result is not None else time.monotonic() + FAILED_LOOKUP_TTL_S
    with _tool_dir_lock:
        _tool_dir_cache[key] = (result, expiry)
    return result


def clear_tool_dir_cache() -> None:
    """Forget cached ``uv tool dir`` answers (tests; a reinstall elsewhere)."""
    with _tool_dir_lock:
        _tool_dir_cache.clear()


def resolve_uv_tool_command(
    command: str,
    args: list[str],
    *,
    env: Mapping[str, str],
    server_env: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> UvToolExec | None:
    """Map a bare ``uvx <pkg> ...`` to the installed ``<tool_dir>/<pkg>/bin/<pkg>``.

    ``env`` is the environment the child would be spawned with; ``server_env``
    is the server's own configured ``env`` block (checked for ``UV_*``
    overrides); ``cwd`` is the server's working directory. The result's
    ``env`` must be overlaid on ``env`` at spawn. Returns ``None`` whenever the
    original command must be kept; see the module docstring for every rule.
    """
    if sys.platform == "win32" or _opted_out():
        return None
    parsed = _uvx_package(command, args)
    if parsed is None:
        return None
    package, tool_args = parsed
    if any(k.startswith("UV_") and k != "UV_TOOL_DIR" for k in (server_env or {})):
        return None
    uv = _uv_binary(command, env.get("PATH"))
    if uv is None:
        return None
    tool_dir = uv_tool_dir(uv, env, cwd)
    if tool_dir is None:
        return None
    env_root = tool_dir / package
    bin_dir = env_root / "bin"
    binary = bin_dir / package
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        logger.debug("no executable installed uv tool for %r; keeping uvx", package)
        return None
    # ``exists()`` follows the symlink, so a dangling interpreter link (base
    # Python removed) fails here and uvx gets the chance to rebuild the env.
    if not ((bin_dir / "python").exists() and (env_root / "uv-receipt.toml").is_file()):
        logger.debug("installed uv tool %r looks broken; keeping uvx", package)
        return None
    path = env.get("PATH")
    overlay = {
        "PATH": f"{bin_dir}{os.pathsep}{path}" if path else str(bin_dir),
        "UV": uv,
    }
    # Names and paths only — never env values.
    logger.debug("exec installed uv tool %r directly at %s (bypassing uv parent)", package, binary)
    return UvToolExec(str(binary), tool_args, overlay)

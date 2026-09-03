"""Atomic discovery state for the browser bridge daemon.

The state file is both cheap createIf discovery and session-leg authorization.
It therefore stays 0600 under a 0700 directory and is replaced atomically so a
session can never consume a half-written key or port.
"""

from __future__ import annotations

import enum
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from local_operator.paths import config_dir

RUN_DIRNAME = "run/browser"
STATE_FILENAME = "bridge.json"
HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_TIMEOUT_S = 45.0


class BridgeState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pid: int
    port: int
    session_key: str = Field(min_length=32)
    proto: int
    extension_connected: bool = False
    paired: bool = False
    extension_id: str = ""
    browser_name: str = ""
    heartbeat_at: float = Field(default_factory=time.time)
    started_at: float = Field(default_factory=time.time)


def run_dir(root: Path | None = None) -> Path:
    """The run directory, CREATED and locked down. Only writers may call this."""
    directory = (root or config_dir()) / RUN_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def state_path(root: Path | None = None) -> Path:
    """Where the discovery file lives. Pure path arithmetic: creates NOTHING.

    It used to route through :func:`run_dir`, which mkdirs and chmods, so every
    READER and every diagnostic performed a write. That turned the ENOSPC log
    line in ``BridgeService.publish_safely`` into a second ``OSError`` raised
    from inside the handler for the first one: on a fresh config dir with a
    full disk the daemon failed to boot, in precisely the disk-full scenario
    this module is meant to survive. An error path may never perform the
    operation that is failing, and detection may never mutate the filesystem
    (see :func:`read`), so the path is now derived without touching disk and
    only the writer (:func:`publish`) asks for the directory to exist.
    """
    return (root or config_dir()) / RUN_DIRNAME / STATE_FILENAME


def publish(state: BridgeState, root: Path | None = None) -> Path:
    directory = run_dir(root)
    state.heartbeat_at = time.time()
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".bridge.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state.model_dump(mode="json"), handle)
        os.chmod(temporary, 0o600)
        target = directory / STATE_FILENAME
        os.replace(temporary, target)
        return target
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read(root: Path | None = None) -> BridgeState | None:
    """Read without mutating or reaping; detection must have no side effects."""
    try:
        raw: Any = json.loads(state_path(root).read_text(encoding="utf-8"))
        return BridgeState.model_validate(raw)
    except (OSError, ValueError, TypeError):
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def heartbeat_age(current: BridgeState, *, now: float | None = None) -> float:
    """Seconds since the daemon last republished. Negative ages clamp to 0.

    Clock skew (or a state file written by a daemon whose clock ran ahead) must
    never read as "fresher than fresh" and must never render as a negative age
    in diagnostics, so the floor is 0.
    """
    timestamp = time.time() if now is None else now
    return max(0.0, timestamp - current.heartbeat_at)


class Liveness(enum.Enum):
    """What the discovery FILE alone can honestly conclude about the daemon.

    The file heartbeat is a liveness PROXY, and it lies in both directions: it
    goes stale while the daemon is perfectly healthy (the heartbeat writer can
    die on its own — see ``BridgeService._supervise``, or the daemon can be
    SIGSTOPped) and it stays fresh for a few seconds after a daemon is killed.
    Collapsing that into one bool is what made a healthy daemon read as a
    permanent "no": every session silently fell back to cmux while
    ``lop browser status`` — which reads the LIVE ``/health`` socket — kept
    reporting the extension connected, and nothing reconciled the two.

    So the file answers three states, not two, and the caller decides how much
    a given answer is worth paying for:

    - ``ABSENT``  no daemon, or no browser attached. A definite no; never probe.
    - ``FRESH``   heartbeat inside the timeout. A definite yes; never probe.
    - ``STALE``   heartbeat expired but the pid is ALIVE and an extension was
      attached when the file was last written. Genuinely unknown from the file:
      only a socket round-trip can settle it (``bridge_browser_reachable``).
    """

    ABSENT = "absent"
    FRESH = "fresh"
    STALE = "stale"


def liveness(
    root: Path | None = None, *, now: float | None = None
) -> tuple[Liveness, BridgeState | None]:
    """Classify the daemon from the file alone: no socket, no subprocess."""
    current = read(root)
    if current is None or not current.extension_connected or not pid_alive(current.pid):
        return Liveness.ABSENT, current
    if heartbeat_age(current, now=now) <= HEARTBEAT_TIMEOUT_S:
        return Liveness.FRESH, current
    return Liveness.STALE, current


def available(root: Path | None = None, *, now: float | None = None) -> bool:
    """Cheap file-only availability gate: FRESH only, no socket, never probes.

    Deliberately still FILE-ONLY and still false for a stale heartbeat. It
    answers "is the bridge known-good right now", which is the question the
    backend-selection paths ask. A stale-but-alive daemon is acquitted on the
    browser path instead, by :func:`~local_operator.browser_bridge.backend.
    bridge_browser_reachable`, which pays for one bounded probe only when it is
    about to condemn the bridge.

    For TOOL GATING use :func:`advertisable` instead — see why there.
    """
    return liveness(root, now=now)[0] is Liveness.FRESH


def advertisable(root: Path | None = None, *, now: float | None = None) -> bool:
    """Whether the `browser` TOOL should be offered. FRESH or STALE-but-alive.

    Gating is a weaker commitment than execution: advertising the tool only
    promises the agent can ASK, and `execute_browser` still decides — with a
    real socket probe — whether the extension answers, falling back to cmux or
    returning the typed demotion diagnostic. So the gate must not apply the
    stricter :func:`available` test.

    It did, and that is a hole in the RC2 rescue: gating ran the FRESH-only
    check, so on an extension-only host (no cmux, the ordinary configuration
    for the extension) a daemon whose heartbeat writer had died was never
    advertised at all. The tool list is built once per session, so that session
    had NO browser tool for its lifetime, `execute_browser` was never reached,
    the socket probe never ran, and the demotion hint — whose every call site
    is inside `execute_browser` — could not fire. The agent got no fallback and
    no diagnostic: strictly worse than the incident this fixes, since it cannot
    even discover that a healthy daemon is sitting there.

    The CONSTRAINT that made the gate file-only still holds and is respected:
    this is synchronous and runs while constructing EVERY session, so it must
    not block or do unbounded I/O. It does neither. STALE is already
    established by :func:`liveness` from one file read plus ``os.kill(pid, 0)``
    — no socket is opened here and no subprocess is spawned, so the hot path
    keeps its measured sub-millisecond cost. When in doubt this errs toward
    advertising: a tool that explains why it cannot reach the bridge beats a
    tool that silently does not exist.
    """
    return liveness(root, now=now)[0] in (Liveness.FRESH, Liveness.STALE)


def remove(root: Path | None = None) -> None:
    try:
        state_path(root).unlink()
    except OSError:
        pass

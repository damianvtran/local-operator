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
    directory = (root or config_dir()) / RUN_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def state_path(root: Path | None = None) -> Path:
    return run_dir(root) / STATE_FILENAME


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
    """Cheap file-only availability gate used while constructing every session.

    Deliberately still FILE-ONLY and still false for a stale heartbeat: this
    runs while constructing every session and on the `createIf` tool-gating
    path, where a blocking socket probe would tax startup for every session on
    the machine. A stale-but-alive daemon is rescued on the browser path
    instead, by :func:`~local_operator.browser_bridge.backend.
    bridge_browser_reachable`, which pays for one bounded probe only when it is
    about to condemn the bridge.
    """
    return liveness(root, now=now)[0] is Liveness.FRESH


def remove(root: Path | None = None) -> None:
    try:
        state_path(root).unlink()
    except OSError:
        pass

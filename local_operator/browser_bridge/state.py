"""Atomic discovery state for the browser bridge daemon.

The state file is both cheap createIf discovery and session-leg authorization.
It therefore stays 0600 under a 0700 directory and is replaced atomically so a
session can never consume a half-written key or port.
"""

from __future__ import annotations

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


def available(root: Path | None = None, *, now: float | None = None) -> bool:
    """Cheap file-only availability gate used while constructing every session."""
    current = read(root)
    if current is None:
        return False
    timestamp = time.time() if now is None else now
    return (
        current.extension_connected
        and pid_alive(current.pid)
        and timestamp - current.heartbeat_at <= HEARTBEAT_TIMEOUT_S
    )


def remove(root: Path | None = None) -> None:
    try:
        state_path(root).unlink()
    except OSError:
        pass

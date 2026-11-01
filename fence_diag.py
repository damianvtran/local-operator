"""Diagnostic-only pytest plugin for the mesh-restart admission fence.

ONE question, answered with evidence: in the run that fails, do the SHARED
acquirer (`registry.publish` of a busy record → `restart_fence.shared`) and the
EXCLUSIVE acquirer (`restart_fence.exclusive`) lock the SAME inode, and do their
intervals OVERLAP the way the fence claims?

Method: wrap the two primitives every acquisition goes through —
`restart_fence._open` (resolves and opens the permanent lock file) and
`restart_fence._flock` (the kernel call) — plus `os.close`, so a descriptor's
whole life is recorded. Identity is `fstat` on the descriptor ACTUALLY locked
and `stat` of the pathname AT THE SAME MOMENT: a replaced lockfile shows as the
two disagreeing, which a path-only comparison can never see.

FAILURE DISCIPLINE (deliberate, and the opposite of a normal plugin):
  * no write happens while a product lock is held to the point of raising —
    a failed write LATCHES, stops writing, and is reported at teardown, because
    raising inside `restart_fence` would corrupt the very ordering being
    measured;
  * output is capped; exceeding the cap is a REPORTED FAILURE, not a silent
    truncation;
  * every process writes its OWN file (`diag-<pid>.jsonl`), so four xdist
    workers never interleave JSON in one file; merging is a post-run step;
  * configuration is fail-closed at `pytest_configure`, where no product lock is
    held: an unwritable output dir, an empty `FENCE_DIAG_WORKTREE`, or a module
    that resolves outside it refuses the run outright.

NOT CI PARITY for dependencies: the interpreter is whatever runs, and it is
labelled as such. What it does assert, strictly, is SOURCE provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# --- configuration, all of it required -------------------------------------

OUT_DIR = os.environ.get("FENCE_DIAG_OUT", "")
WORKTREE_ENV = os.environ.get("FENCE_DIAG_WORKTREE", "")
MAX_EVENTS = int(os.environ.get("FENCE_DIAG_MAX_EVENTS", "20000"))

_MODULES = (
    "local_operator",
    "local_operator.network.meshrestart",
    "local_operator.session.runtime.restart_fence",
    "local_operator.session.runtime.registry",
)

_state = {
    "lost": False,          # a write failed; stop writing, report at teardown
    "truncated": False,     # cap reached; stop writing, report at teardown
    "events": 0,
    "path": None,
}
_fd_paths: dict[int, str] = {}
_tracked: set[int] = set()


def _require_config() -> tuple[Path, Path]:
    if not OUT_DIR:
        raise RuntimeError("FENCE_DIAG_OUT must name an output DIRECTORY")
    if not WORKTREE_ENV:
        raise RuntimeError("FENCE_DIAG_WORKTREE must name the exact source worktree")
    worktree = Path(WORKTREE_ENV).resolve()
    if not worktree.is_dir():
        raise RuntimeError(f"FENCE_DIAG_WORKTREE does not exist: {worktree}")
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out, worktree


def _write(entry: dict[str, Any]) -> None:
    """Append one record. Never raises: a diagnostic must not break the subject."""
    if _state["lost"] or _state["truncated"]:
        return
    if _state["events"] >= MAX_EVENTS:
        _state["truncated"] = True
        return
    record = {"t": round(time.time(), 6), "mono": round(time.monotonic(), 6),
              "pid": os.getpid(), "thread": threading.current_thread().name}
    record.update(entry)
    try:
        with open(_state["path"], "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        _state["events"] += 1
    except OSError:
        _state["lost"] = True


def _blob_digests(worktree: Path) -> dict[str, str]:
    import importlib

    digests: dict[str, str] = {}
    for name in _MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if worktree not in path.parents:
            raise RuntimeError(f"PROVENANCE FAIL: {name} resolved to {path}, not under {worktree}")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return digests


def _lock_path_for(root: Any) -> str:
    """The path `restart_fence._open` itself would resolve — not an invented one.

    `_open` computes `(root or config_dir()).expanduser().resolve()`; this mirrors
    that expression exactly, reading `config_dir` from the module that calls it.
    """
    from local_operator.session.runtime import restart_fence

    base = Path(root) if root else restart_fence.config_dir()
    return str((base.expanduser().resolve()) / "run" / restart_fence.FENCE_NAME)


def _identity(descriptor: int, root: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"fd": descriptor}
    try:
        info = os.fstat(descriptor)
        out.update({"fd_st_dev": info.st_dev, "fd_st_ino": info.st_ino})
    except OSError as exc:
        out["fd_fstat_error"] = f"{type(exc).__name__}: {exc}"
    path = _fd_paths.get(descriptor) or _lock_path_for(root)
    out["lock"] = path
    try:
        named = Path(path).stat()
        out.update({"path_st_dev": named.st_dev, "path_st_ino": named.st_ino})
    except OSError as exc:
        out["path_stat_error"] = f"{type(exc).__name__}: {exc}"
    if "fd_st_ino" in out and "path_st_ino" in out:
        out["inode_matches"] = (out["fd_st_dev"], out["fd_st_ino"]) == (
            out["path_st_dev"], out["path_st_ino"])
    return out


def _install() -> None:
    from local_operator.session.runtime import restart_fence

    if getattr(restart_fence._open, "_fence_diag", False):
        return

    real_open = restart_fence._open
    real_flock = restart_fence._flock
    real_close = os.close

    def observed_open(root: Any) -> int:
        descriptor = real_open(root)
        try:
            _fd_paths[descriptor] = _lock_path_for(root)
            _tracked.add(descriptor)
            _write({"event": "open", "root_repr": repr(root), **_identity(descriptor, root)})
        except Exception:  # noqa: BLE001 — diagnostics never break the subject
            _state["lost"] = True
        return descriptor

    def observed_flock(descriptor: int, operation: int) -> None:
        try:
            _write({"event": "flock.enter", "op": operation, **_identity(descriptor, None)})
        except Exception:  # noqa: BLE001
            _state["lost"] = True
        started = time.monotonic()
        try:
            real_flock(descriptor, operation)
        except BaseException as exc:
            try:
                _write({"event": "flock.raised", "op": operation,
                        "held_s": round(time.monotonic() - started, 6),
                        "error": f"{type(exc).__name__}: {exc}",
                        **_identity(descriptor, None)})
            except Exception:  # noqa: BLE001
                _state["lost"] = True
            raise
        try:
            _write({"event": "flock.returned", "op": operation,
                    "blocked_s": round(time.monotonic() - started, 6),
                    **_identity(descriptor, None)})
        except Exception:  # noqa: BLE001
            _state["lost"] = True

    def observed_close(fd: int) -> None:
        if fd in _tracked:
            try:
                _write({"event": "close", **_identity(fd, None)})
            except Exception:  # noqa: BLE001
                _state["lost"] = True
            _tracked.discard(fd)
            _fd_paths.pop(fd, None)
        real_close(fd)

    setattr(observed_open, "_fence_diag", True)
    setattr(observed_flock, "_fence_diag", True)
    restart_fence._open = observed_open  # type: ignore[assignment]
    restart_fence._flock = observed_flock  # type: ignore[assignment]
    os.close = observed_close  # type: ignore[assignment]
    sys.modules["os"].close = observed_close  # type: ignore[assignment]


# --- pytest hooks ----------------------------------------------------------

def pytest_configure(config: Any) -> None:
    for arg in config.invocation_params.args:
        if str(arg) in ("-P", "--import-mode=importlib"):
            raise RuntimeError(f"diagnostic refuses {arg}: it changes import resolution")
    out, worktree = _require_config()
    _state["path"] = out / f"diag-{os.getpid()}.jsonl"
    # FAIL-CLOSED SETUP: prove the sink works before the run, while no product
    # lock is held. A read-back, not just a successful open.
    probe = {"event": "setup", "blobs": _blob_digests(worktree), "worktree": str(worktree),
             "worker": hasattr(config, "workerinput"), "pid": os.getpid(),
             "python": sys.version.split()[0], "max_events": MAX_EVENTS}
    with open(_state["path"], "a") as handle:
        handle.write(json.dumps(probe, sort_keys=True) + "\n")
    readback = Path(_state["path"]).read_text().strip().splitlines()
    if not readback or json.loads(readback[-1]).get("event") != "setup":
        raise RuntimeError(f"FENCE_DIAG setup read-back failed at {_state['path']}")
    _state["events"] = 1
    _install()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    summary = {"event": "summary", "pid": os.getpid(), "exitstatus": int(exitstatus),
               "events": _state["events"], "lost": _state["lost"],
               "truncated": _state["truncated"], "max_events": MAX_EVENTS,
               "out": str(_state["path"])}
    try:
        summary_path = Path(_state["path"]).with_name(f"summary-{os.getpid()}.jsonl")
        with open(summary_path, "w") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    except OSError as exc:
        raise RuntimeError(f"FENCE_DIAG could not write its summary: {exc}") from exc
    # A diagnostic that lost or truncated its evidence must never look green.
    if _state["lost"] or _state["truncated"]:
        raise RuntimeError(
            f"FENCE_DIAG evidence incomplete (lost={_state['lost']}, "
            f"truncated={_state['truncated']}, events={_state['events']}, "
            f"cap={MAX_EVENTS}): treat this run as BLOCKED, not passed")

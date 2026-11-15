"""The mobile daemon's open-file limit: raise it at start, and say what it is.

WHY THIS EXISTS. A teammate's phone showed "Asleep / Not answering" while his
Mac reported the daemon healthy, beside sporadic sqlite "unable to open database
file" errors from the attention store. Both are what a process at its
descriptor ceiling looks like: every SSE stream the phone holds, every session
socket and every sqlite open is one descriptor, and the daemon's LaunchAgent
(`install.render_plist`) sets no `SoftResourceLimits`, so launchd starts it at
the macOS per-job default soft limit of 256 — low enough for a busy phone plus
a few dozen sessions to reach. systemd's user default (1024) is higher but
still finite.

The fix is in-process rather than in the plist so ONE change covers launchd,
systemd, Task Scheduler and a foreground `lop mobile serve`, and so an
existing install picks it up on its next restart without a plist rewrite.

THE CAP (`TARGET_SOFT_LIMIT`, 4096) must sit above the daemon's observed
descriptor use; the measurement backing it (n/median/max on a rig under load)
is recorded in the PR that introduced this module. It is a SOFT limit only,
never above the hard limit, so it needs no privilege and cannot fail a start:
macOS also refuses a soft limit above `kern.maxfilesperproc` (OPEN_MAX 10240)
when the hard limit is infinite, which 4096 stays below.

The daemon records what it ended up with in `<config>/mobile/daemon-limits.json`
(pid, soft, hard, before), because a status reader in ANOTHER process cannot
read a process's limits portably — Linux has `/proc/<pid>/limits`, macOS has no
equivalent short of a debugger. The pid in the record is what lets a reader
discard a stale file left by a previous daemon.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir

TARGET_SOFT_LIMIT = 4096


def limits_path() -> Path:
    return config_dir() / "mobile" / "daemon-limits.json"


def raise_soft_limit(target: int = TARGET_SOFT_LIMIT) -> dict[str, int | None]:
    """Raise RLIMIT_NOFILE's soft limit to ``min(hard, target)``; never lower it.

    Returns ``{"before", "soft", "hard"}`` (``hard`` is ``None`` for unlimited).
    Never raises: a platform without `resource` (Windows) or a refused call
    leaves the limit as it was, and the daemon starts exactly as before.
    """
    try:
        import resource
    except ImportError:  # Windows: no RLIMIT_NOFILE to raise.
        return {"before": None, "soft": None, "hard": None}
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    before = soft
    ceiling = target if hard == resource.RLIM_INFINITY else min(hard, target)
    if soft != resource.RLIM_INFINITY and soft < ceiling:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (ceiling, hard))
            soft = ceiling
        except (ValueError, OSError):
            pass
    infinite = resource.RLIM_INFINITY
    return {
        "before": None if before == infinite else int(before),
        "soft": None if soft == infinite else int(soft),
        "hard": None if hard == infinite else int(hard),
    }


def record(limits: dict[str, int | None], *, pid: int | None = None) -> None:
    """Publish this daemon's limits for status readers (best-effort)."""
    path = limits_path()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"pid": pid or os.getpid(), **limits}) + "\n")
        os.replace(temporary, path)
    except OSError:
        pass


def recorded(pid: int | None) -> dict[str, Any] | None:
    """The limits the RUNNING daemon recorded, or ``None`` when stale/absent.

    A record whose pid is not ``pid`` describes a previous daemon and is
    discarded rather than reported as this one's ceiling.
    """
    try:
        value = json.loads(limits_path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or pid is None or value.get("pid") != pid:
        return None
    return value

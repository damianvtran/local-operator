"""Remember which terminal the user last sat in, so a notification click can reopen it.

WHY THIS EXISTS. The last rung of the click ladder (``tui.resume_click``) runs in
a process that, by construction, is NOT inside any terminal: a notification
click is handled by a detached helper whose environment carries none of the
markers :func:`local_operator.spawn.registry.active_backend` keys on. Detection
therefore answers "nothing" there, and the rung fell through to a hard-coded
Terminal.app on macOS — and to nothing at all elsewhere. A person who lives in
Ghostty clicked a banner and got a Terminal.app window they never use.

There is no "default terminal" setting on macOS or in this project to ask, and
inventing one would be a preference the user never stated. What they DID state,
by using it, is the terminal they last ran ``lop`` in. So an ATTENDED moment —
the TUI booting inside an emulator, where detection works — writes the winning
backend's name here, and the click, which cannot detect, recalls it.

WHAT IS STORED, AND WHAT IS NOT. Only the backend ``name`` (``"ghostty"``,
``"kitty"``, ``"iterm2"``…) and a timestamp: no path, no cwd, no environment.
The backends are stateless by design (``spawn/types.py``) and resolve their own
binary at spawn time, so a name is a complete pointer and cannot go stale in a
way that matters — an uninstalled terminal makes that backend's ``spawn`` answer
False and the ladder moves on to the next candidate.

Best-effort in both directions, like everything on this path: an unwritable
config dir or a torn file is "no memory", never an exception.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FILE = "last-terminal.json"


def _path(config_dir: Path) -> Path:
    # Beside the notifier bundle: both exist only to make a banner's click work,
    # and the directory is already per-user and writable.
    return config_dir / "notifier" / _FILE


#: Backends whose "window" is not something a click can SHOW the user.
#: ``CmuxBackend.spawn`` passes ``--focus false`` on every call (its module
#: docstring: a fork must not steal the window being typed in), which is right
#: for ``/fork`` and wrong for a click: the user would get an unfocused sidebar
#: row in a cmux window that may be on another Space, i.e. a click that appears
#: to do nothing. Not remembered, so a cmux user's click keeps the existing
#: Terminal.app landing, which at least raises a window. Enforced on the READ as
#: well as the write (:func:`recall`), so a hand-edited or older file naming one
#: is not honoured.
_NOT_REMEMBERED = frozenset({"cmux"})

#: How long a memory is believed. THE MEMORY HAS TO BE ABLE TO BE WRONG-AND-GONE,
#: because the thing it stands for ("the terminal the user is at") changes
#: without this file being told: someone who tried Ghostty once and now lives in
#: an emulator we cannot name (VS Code's terminal, Alacritty, Warp) would
#: otherwise get a Ghostty window on every click, forever (found in review, D2).
#: Two mechanisms, because each covers a hole in the other:
#:
#: - **Clear on an attended boot that cannot name its terminal**
#:   (:func:`remember_current` with ``forget_if_undetected``) — immediate and
#:   exact for the user who moved, but silent about the user who simply stopped
#:   booting a TUI at all.
#: - **Age out** — the backstop for that second user. The TUI refreshes
#:   ``recorded_at`` once it is older than :data:`_REFRESH_AFTER_S`, so a daily
#:   Ghostty user never reaches the limit and a user who has not booted a TUI in
#:   a month is not sent to a terminal they may have uninstalled.
MAX_AGE_S = 30 * 24 * 3600

#: A write is skipped while the file already says the same thing and is younger
#: than this, so a TUI booting many times a day in one terminal costs a read.
_REFRESH_AFTER_S = 12 * 3600


def _read(config_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(_path(config_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def recall(config_dir: Path, *, now: float | None = None) -> str | None:
    """The backend name last remembered, or None.

    None for: no file, a torn file, a wrong shape, a name in
    :data:`_NOT_REMEMBERED`, or a record older than :data:`MAX_AGE_S` (or with
    no usable ``recorded_at`` — an undated memory cannot be shown to be fresh).
    """
    data = _read(config_dir)
    if data is None:
        return None
    name = data.get("backend")
    if not isinstance(name, str) or not name or name in _NOT_REMEMBERED:
        return None
    recorded = data.get("recorded_at")
    if not isinstance(recorded, (int, float)) or isinstance(recorded, bool):
        return None
    if (time.time() if now is None else now) - recorded > MAX_AGE_S:
        return None
    return name


def remember(config_dir: Path, backend_name: str, *, now: float | None = None) -> bool:
    """Record ``backend_name`` as the terminal last attended. True if written."""
    if not backend_name or backend_name in _NOT_REMEMBERED:
        return False
    stamp = time.time() if now is None else now
    data = _read(config_dir)
    if (
        data is not None
        and recall(config_dir, now=stamp) == backend_name
        and isinstance(data.get("recorded_at"), (int, float))
        and stamp - data["recorded_at"] < _REFRESH_AFTER_S
    ):
        return False
    path = _path(config_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp + replace: two TUIs booting together must not leave a
        # half-written file for a click to misread.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(
            json.dumps({"backend": backend_name, "recorded_at": int(stamp)}),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return True
    except OSError:
        logger.debug("could not remember the attended terminal", exc_info=True)
        return False


def forget(config_dir: Path) -> bool:
    """Drop the memory. True if a file was removed."""
    try:
        _path(config_dir).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.debug("could not forget the attended terminal", exc_info=True)
        return False


def remember_current(
    config_dir: Path,
    env: Mapping[str, str] | None = None,
    *,
    forget_if_undetected: bool = False,
) -> bool:
    """Record the terminal THIS process is running in, if one is detectable.

    Call from an ATTENDED moment (the TUI booting), where emulator markers are
    present. Never raises; returns True only when the file changed.

    ``forget_if_undetected`` is for a caller that IS a person's terminal (the
    TUI): if it cannot name the emulator it is in (or it is cmux, which is never
    remembered), then the user is demonstrably no longer in the terminal the file
    names, and keeping it would send their next click there. A caller that is
    NOT a terminal — the desktop daemon, which has no emulator markers because it
    has no emulator — must leave this False, or every daemon boot would wipe a
    perfectly good memory.
    """
    try:
        from local_operator.spawn.registry import active_backend

        backend = active_backend(env)
    except Exception:  # noqa: BLE001 — remembering is a nicety
        logger.debug("could not detect the attended terminal", exc_info=True)
        return False
    if backend is None or backend.name in _NOT_REMEMBERED:
        return forget(config_dir) if forget_if_undetected else False
    return remember(config_dir, backend.name)

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

logger = logging.getLogger(__name__)

_FILE = "last-terminal.json"


def _path(config_dir: Path) -> Path:
    # Beside the notifier bundle: both exist only to make a banner's click work,
    # and the directory is already per-user and writable.
    return config_dir / "notifier" / _FILE


def remember(config_dir: Path, backend_name: str) -> bool:
    """Record ``backend_name`` as the terminal last attended. True if written.

    Skips the write when the file already says the same thing, so a TUI that
    boots many times a day in one terminal costs a read, not a rewrite.
    """
    if not backend_name:
        return False
    if recall(config_dir) == backend_name:
        return False
    path = _path(config_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp + replace: two TUIs booting together must not leave a
        # half-written file for a click to misread.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(
            json.dumps({"backend": backend_name, "recorded_at": int(time.time())}),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return True
    except OSError:
        logger.debug("could not remember the attended terminal", exc_info=True)
        return False


def recall(config_dir: Path) -> str | None:
    """The backend name last remembered, or None (absent, torn, wrong shape)."""
    try:
        data = json.loads(_path(config_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("backend") if isinstance(data, dict) else None
    return name if isinstance(name, str) and name else None


#: Backends whose "window" is not something a click can SHOW the user.
#: ``CmuxBackend.spawn`` passes ``--focus false`` on every call (its module
#: docstring: a fork must not steal the window being typed in), which is right
#: for ``/fork`` and wrong for a click: the user would get an unfocused sidebar
#: row in a cmux window that may be on another Space, i.e. a click that appears
#: to do nothing. Not remembered, so a cmux user's click keeps the existing
#: Terminal.app landing, which at least raises a window.
_NOT_REMEMBERED = frozenset({"cmux"})


def remember_current(config_dir: Path, env: Mapping[str, str] | None = None) -> bool:
    """Record the terminal THIS process is running in, if one is detectable.

    Call from an ATTENDED moment (the TUI booting), where emulator markers are
    present. Never raises; returns True only when the file changed.
    """
    try:
        from local_operator.spawn.registry import active_backend

        backend = active_backend(env)
    except Exception:  # noqa: BLE001 — remembering is a nicety
        logger.debug("could not detect the attended terminal", exc_info=True)
        return False
    if backend is None or backend.name in _NOT_REMEMBERED:
        return False
    return remember(config_dir, backend.name)

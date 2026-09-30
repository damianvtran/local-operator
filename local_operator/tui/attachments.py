"""Hand one stored attachment to the platform's own opener (spec §7.4).

The detail page shows an attachment's path and never pretends to render the
file; `↵` on the row asks the OS to open the COPY the store keeps beside the
update. Two constraints shape the mechanism, and both are the reason this is
not a bare ``subprocess`` call:

* **The opener is per PLATFORM, and absent where none exists.** macOS ships
  ``open``; Linux desktops ship ``xdg-open``. Anything else gets ``None`` — a
  wrong guess would spawn whatever happened to be on ``PATH``, and an honest
  refusal is what the caller can turn into a sentence.
* **The child's streams are pipes this process owns.** Both openers write to
  stderr (``xdg-open: no method available``, LaunchServices complaints), and a
  raw spawn INHERITS the terminal, so that chatter lands in the middle of a
  Textual frame. This is the same defect ``open_browser_quietly`` exists to
  prevent, solved the same way: read the pipe, log it, show nothing.

Nothing here raises: a platform with no opener, a missing binary and a failing
opener are all ``False``, which the host answers with the honest sentence.
"""

from __future__ import annotations

import asyncio
import sys

from local_operator.logger import get_logger

logger = get_logger(__name__)

#: How long an opener may run before the call is abandoned. Openers hand off to
#: a GUI and return; one that hangs is a wedged desktop, not a slow file, so the
#: bound is short and its expiry is a plain ``False`` (spec §7.4's honest path).
OPEN_PATH_TIMEOUT_S = 10.0


def opener_argv(path: str) -> list[str] | None:
    """The platform's opener for one path, or ``None`` where there is none."""
    if sys.platform == "darwin":
        return ["open", path]
    if sys.platform.startswith("linux"):
        return ["xdg-open", path]
    return None


async def open_path_quietly(path: str) -> bool:
    """Open ``path`` with the OS opener. ``False`` when it cannot or did not.

    A ``False`` never means "maybe" — it means nothing was launched, or the
    launcher reported a non-zero exit. The caller states that outcome rather
    than swallowing it, which is the whole point of returning a boolean here.

    ONE deadline covers the whole call: the read and the wait must not each
    spend :data:`OPEN_PATH_TIMEOUT_S`, or a wedged opener is killed at twice
    the bound the constant states (measured 20.0 s against a ``sleep 30``
    child — agent review round 1, MINOR-3).
    """
    argv = opener_argv(path)
    if argv is None:
        logger.debug("no attachment opener on this platform (%s)", sys.platform)
        return False
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            # Merged: the two streams are one diagnostic here, and a single pipe
            # cannot deadlock against itself the way two unread ones can.
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception:  # noqa: BLE001 — a missing opener is a degraded row, not a crash
        logger.debug("attachment opener failed to start: %s", argv[0], exc_info=True)
        return False

    async def _read_and_wait() -> int:
        if process.stdout is not None:
            raw = await process.stdout.read()
            text = raw.decode("utf-8", "replace").strip()
            if text:
                logger.info("attachment opener (%s): %s", argv[0], text)
        return await process.wait()

    try:
        return await asyncio.wait_for(_read_and_wait(), timeout=OPEN_PATH_TIMEOUT_S) == 0
    except asyncio.TimeoutError:
        # The opener is named rather than this process's own entry path: the
        # message is about the child that hung (agent review round 1, MINOR-2).
        logger.debug("attachment opener timed out: %s", argv[0])
        try:
            process.kill()
        except ProcessLookupError:  # pragma: no cover — it exited between the two
            pass
        return False
    except Exception:  # noqa: BLE001 — a drain failure must not mask the exit code
        logger.debug("attachment opener failed", exc_info=True)
        return False

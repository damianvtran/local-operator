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

The module also owns the READ half of the inline preview (spec §7.4's staged
enhancement): :func:`read_for_preview` turns a stored copy into the base64 the
transcript's own :class:`~local_operator.tui.widgets.image_block.ImageBlock`
takes. Keeping it here rather than in the app is the same boundary the opener
respects — the widget layer never touches the filesystem, and the app never
re-implements how an attachment is addressed.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import sys
from pathlib import Path

from local_operator.logger import get_logger
from local_operator.projects import ATTACHMENT_MAX_BYTES

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


#: How long a preview read may take before it is abandoned. A stored attachment
#: is at most 5 MB (``projects.ATTACHMENT_MAX_BYTES``) on a local disk, so this
#: bound exists for the pathological case — a copy on a stalled network mount —
#: and its expiry is a plain ``None`` the caller turns into a sentence.
PREVIEW_READ_TIMEOUT_S = 5.0


def read_attachment_for_preview(path: str) -> tuple[str, str] | None:
    """The base64 payload and MIME type for one stored copy, or ``None``.

    ``None`` covers every way there is nothing to show — the path is gone, it
    is a directory, or the read failed — because the caller states ONE honest
    sentence for all three ("could not read"). The MIME type falls back to
    ``image/png`` only when the suffix is unknown: the block decodes the bytes
    itself, so a wrong guess shows the ``could not be decoded`` receipt rather
    than a wrong picture (and an ``.svg``, which PIL cannot rasterise, lands
    there on purpose).

    Synchronous on purpose: the caller runs it off the UI loop, and a plain
    function is what makes that a ``to_thread`` hop rather than an event-loop
    read that blocks the frame.

    The store's own size cap is re-checked HERE rather than assumed: it is
    enforced at write time (``projects._store_attachments``), so the only way
    past it is a hand-edited record or a file replaced after the copy — and
    either would otherwise be read whole into memory (base64 is ~1.37x it, plus
    the decoded frame). Over the cap is ``None``, the same honest answer as
    unreadable (agent review round 1, M1).
    """
    try:
        source = Path(path)
        if not source.is_file():
            return None
        if source.stat().st_size > ATTACHMENT_MAX_BYTES:
            logger.debug("attachment preview refused: %s is over the store's cap", path)
            return None
        data = source.read_bytes()
    except Exception:  # noqa: BLE001 — an unreadable copy is a degraded row, not a crash
        logger.debug("attachment preview read failed: %s", path, exc_info=True)
        return None
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    return base64.b64encode(data).decode("ascii"), mime_type


async def read_for_preview(path: str) -> tuple[str, str] | None:
    """Off-loop wrapper around :func:`read_attachment_for_preview`.

    The read happens on a worker thread so a slow or stalled mount cannot hold
    a Textual frame, and the whole thing is bounded so it cannot hold one
    forever either.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(read_attachment_for_preview, path),
            timeout=PREVIEW_READ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.debug("attachment preview read timed out: %s", path)
        return None

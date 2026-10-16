"""The key-free TTS rung: macOS ``say`` (``local_tts_system``, design V9/V10).

It needs no credential and no download, so voice mode works on a Mac with no
provider signed in. The design measured 0.82-1.38 s to render a 5.0 s clip at
a load average of 4-6.

WAV ONLY, and that is why the rung is OPT-IN. ``say`` writes AIFF, CAF, WAV,
M4A and similar containers but never MP3, and every existing caller of the
cascade (``POST /v1/tools/speech``) labels its answer ``audio/{response_format}``
with mp3 as the default. Serving WAV bytes under an mp3 label would break that
caller. Only the voice routes pass ``include_local=True``, and they sniff the
container they actually got. ``GET /v1/tts/paths`` therefore reports exactly
what it reported before.

The text goes in through a FILE (``-f``), never argv: a reply that begins with
``-`` must not be read as an option, and argv has a length limit.

The child runs in its own process group and the whole group is killed on
timeout or cancellation, so a wedged speech engine cannot outlive the request.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

#: The provider label the spend record and the outcome carry for this rung.
PROVIDER = "macos_say"

#: 16-bit little-endian PCM at 22.05 kHz: the smallest WAV that sounds like the
#: system voice, and a container every renderer's ``<audio>`` element can play.
_DATA_FORMAT = "LEI16@22050"

AVAILABLE_REASON = "The system voice (macOS say) is available."
UNAVAILABLE_REASON = "The system voice is unavailable: macOS say was not found."


def say_binary() -> str | None:
    """The ``say`` executable, or ``None`` off macOS or when it is missing."""
    if sys.platform != "darwin":
        return None
    return shutil.which("say")


def available() -> bool:
    return say_binary() is not None


class LocalSayError(RuntimeError):
    """``say`` ran and failed, or produced no audio."""


async def synthesize(text: str, *, timeout_s: float) -> bytes:
    """Render ``text`` to WAV bytes with the system voice.

    Raises :class:`LocalSayError` on a failed or empty render, and
    ``asyncio.TimeoutError`` when the bound expires. The cascade turns both into
    a failed attempt.
    """
    binary = say_binary()
    if binary is None:
        raise LocalSayError(UNAVAILABLE_REASON)
    with tempfile.TemporaryDirectory(prefix="lop-say-") as directory:
        source = Path(directory) / "input.txt"
        target = Path(directory) / "speech.wav"
        source.write_text(text, encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            binary,
            "-o",
            str(target),
            "--file-format=WAVE",
            f"--data-format={_DATA_FORMAT}",
            "-f",
            str(source),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except BaseException:
            _kill_group(process.pid)
            await process.wait()
            raise
        if process.returncode != 0:
            detail = (stderr or b"").decode("utf-8", "replace").strip()[:200]
            raise LocalSayError(f"say exited with status {process.returncode}: {detail}")
        try:
            audio = target.read_bytes()
        except OSError as error:
            raise LocalSayError(f"say produced no audio file: {error}") from error
    if not audio:
        raise LocalSayError("say produced an empty audio file")
    return audio


def _kill_group(pid: int) -> None:
    """Kill the child's process group, which ``start_new_session`` made its own."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

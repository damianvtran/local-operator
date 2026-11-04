"""The detached notification-sound player: ``python -m local_operator.notifications.play``.

WHY A CHILD PROCESS. Playback takes the sound's whole duration (0.3-1.5 s), and
the callers are a TUI event loop and a runtime's gate path, which may not
block. A thread in a long-lived process would also keep an audio device open
for the life of the session. A detached child (``proc.spawn_detached``) costs
50-70 ms of wall time from spawn to decoded-and-ready (measured in the design,
§4.2), opens the device only while the sound plays, and exits.

WHY MINIAUDIO FIRST. It is a base dependency, decodes FLAC on every OS and has
native backends for CoreAudio, PulseAudio/ALSA and WASAPI. One implementation
therefore covers Windows too, where ``detached_notify`` raises no banner at all
but the TUI's in-band banner still deserves a sound. ``afplay`` is the macOS
fallback for an install whose miniaudio wheel cannot open a device. It takes
``-v`` as a linear gain, the same unit as ``--gain``.

THE GAIN IS APPLIED TO THE SAMPLES, not to a device or system volume. The
system volume stays the user's, and lop's slider scales within it. The decode
is 16-bit, and so is the scaled output; at the default gain of 0.49 the
requantisation noise sits below -90 dBFS, under the source's own dither.

TEST AND QA SEAM. ``LOCAL_OPERATOR_NO_SOUND`` set: exit 0 without touching a
device. ``LOCAL_OPERATOR_SOUND_BACKEND=null``: run the whole decode-and-play
path on miniaudio's NULL device, which consumes samples at real time with no
speakers, then append a receipt line ``{sound_id, sha256, gain, leg, backend,
frames, t}`` to ``<config>/run/notification-sounds.jsonl``. The receipt is
written AFTER playback, so it proves the path ran to the end, not just that a
child was started. A real-speaker run writes no receipt: the jsonl file is a
QA artifact, not a log a user's machine accumulates.

Exit codes: 0 played (or muted), 1 nothing could play the file, 2 usage.
"""

from __future__ import annotations

import argparse
import array
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

#: How long past the sound's own duration to keep the device open, so the
#: device's last buffer drains before the stream is closed. Without it the tail
#: (the fade the masters were cut with) can be clipped off.
_DRAIN_S = 0.25

#: Headroom past the sound's own duration before the child gives up on a hung
#: device or ``afplay`` (review round 1, m5). A wedged CoreAudio or PulseAudio
#: daemon can block a device open or start forever, and one more child would
#: start per notification after each throttle window, so every child must exit
#: on its own. Five seconds is far past any healthy open (tens of ms).
_HANG_MARGIN_S = 5.0
#: The bound when the duration cannot even be read (the file then fails to
#: decode anyway, so this is a backstop, not a budget).
_UNKNOWN_DURATION_S = 30.0

_ENV_NO_SOUND = "LOCAL_OPERATOR_NO_SOUND"
_ENV_BACKEND = "LOCAL_OPERATOR_SOUND_BACKEND"


def _scaled(samples: "array.array[int]", gain: float) -> "array.array[int]":
    """``samples`` (signed 16-bit) times ``gain``.

    One comprehension into a fresh array, not an indexed loop (review round 1,
    n2): about 5x faster in CPython, roughly 15 ms for a 0.8 s stereo sound.
    numpy would do better but is not a dependency, and this child must not add
    one. ``gain`` is clamped to [0, 1] by every caller, so ``|value * gain|``
    never exceeds 32768 and no per-sample range clamp is needed; ``round``
    keeps the requantisation symmetric about zero.
    """
    if gain >= 0.9999:
        return samples
    return array.array("h", [round(value * gain) for value in samples])


def _hard_deadline(path: Path) -> float:
    """Seconds after which this child must exit, whatever it is blocked on."""
    try:
        import miniaudio

        duration = float(miniaudio.get_file_info(str(path)).duration)
    except Exception:  # noqa: BLE001 — unreadable file: use the backstop
        duration = _UNKNOWN_DURATION_S
    return duration + _DRAIN_S + _HANG_MARGIN_S


def _arm_watchdog(seconds: float) -> threading.Timer:
    """Hard-exit the process after ``seconds``.

    ``os._exit`` from a daemon timer rather than an exception or a signal. The
    thing that hangs is a C call inside miniaudio (device open, start or
    close), and neither an exception raised in another thread nor ``SIGALRM``
    (main thread only, absent on Windows) can interrupt it. Exiting the process
    is the one action that always works, and the OS releases the device.
    """
    timer = threading.Timer(seconds, os._exit, args=(3,))
    timer.daemon = True
    timer.start()
    return timer


def _play_miniaudio(path: Path, gain: float, *, null_backend: bool) -> int:
    """Play ``path`` through miniaudio and return the number of frames played.

    Raises on any decode or device failure so :func:`main` can fall back.
    """
    import miniaudio

    decoded = miniaudio.decode_file(str(path), nchannels=2, sample_rate=44100)
    samples = _scaled(decoded.samples, gain)
    frames = len(samples) // 2
    backends = [miniaudio.Backend.NULL] if null_backend else None

    def feed():  # type: ignore[no-untyped-def]
        # miniaudio's generator protocol: it sends the number of frames it
        # wants and takes back that many interleaved samples, or fewer at the end.
        position = 0
        wanted = yield b""
        while position < len(samples):
            chunk = samples[position : position + wanted * 2]
            position += len(chunk)
            wanted = yield chunk
        # Keep yielding silence until the device is stopped, so the last real
        # buffer is not cut off by a StopIteration mid-period.
        while True:
            wanted = yield array.array("h", bytes(wanted * 4))

    generator = feed()
    next(generator)
    with miniaudio.PlaybackDevice(
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=2,
        sample_rate=44100,
        backends=backends,
        app_name="Local Operator",
    ) as device:
        device.start(generator)
        time.sleep(frames / 44100 + _DRAIN_S)
    return frames


def _play_afplay(path: Path, gain: float, timeout_s: float = _UNKNOWN_DURATION_S) -> bool:
    afplay = shutil.which("afplay") if sys.platform == "darwin" else None
    if not afplay:
        return False
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [afplay, "-v", f"{gain:.4f}", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            # `run` kills the child on expiry, so a hung afplay cannot outlive
            # this player (m5).
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _write_receipt(record: dict[str, object]) -> None:
    from local_operator.notifications.sound import RECEIPT_FILE, run_dir

    target = run_dir() / RECEIPT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    # One write() of one line in append mode: concurrent players cannot
    # interleave inside a line.
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def play(
    path: Path,
    gain: float,
    *,
    sound_id: str = "",
    sha256: str = "",
    leg: str = "",
    env: dict[str, str] | None = None,
    arm_watchdog: bool = False,
) -> int:
    """Play once, synchronously. The importable core of the CLI below.

    ``arm_watchdog`` is for the detached child only (:func:`main`): its timer
    ``os._exit``s the whole process, which is right for a child that exists to
    play one sound and wrong for ``lop sound preview`` or a test calling this
    in-process. Those still get the ``afplay`` timeout.
    """
    source = os.environ if env is None else env
    if source.get(_ENV_NO_SOUND):
        return 0
    null_backend = source.get(_ENV_BACKEND, "").strip().lower() == "null"
    if gain <= 0.0:
        return 0
    deadline = _hard_deadline(path)
    watchdog = _arm_watchdog(deadline + 1.0) if arm_watchdog else None
    try:
        return _play_once(path, gain, sound_id, sha256, leg, null_backend, deadline)
    finally:
        if watchdog is not None:
            watchdog.cancel()


def _play_once(
    path: Path,
    gain: float,
    sound_id: str,
    sha256: str,
    leg: str,
    null_backend: bool,
    deadline: float,
) -> int:
    try:
        frames = _play_miniaudio(path, gain, null_backend=null_backend)
        backend = "null" if null_backend else "miniaudio"
    except Exception:  # noqa: BLE001 — any miniaudio failure falls back
        if null_backend:
            # The NULL seam must never reach real speakers: a QA rig asked for
            # silence, so a failure here is reported, not "fixed" by afplay.
            return 1
        return 0 if _play_afplay(path, gain, deadline) else 1
    if null_backend:
        _write_receipt(
            {
                "sound_id": sound_id,
                "sha256": sha256,
                "gain": round(gain, 4),
                "leg": leg,
                "backend": backend,
                "frames": frames,
                "t": round(time.time(), 3),
            }
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Developer-facing argv for a child nobody types; English by design.
    parser = argparse.ArgumentParser(prog="local_operator.notifications.play")
    parser.add_argument("path", type=Path)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--sound-id", default="")
    parser.add_argument("--sha256", default="")
    parser.add_argument("--leg", default="")
    args = parser.parse_args(argv)
    return play(
        args.path,
        min(1.0, max(0.0, args.gain)),
        sound_id=args.sound_id,
        sha256=args.sha256,
        leg=args.leg,
        arm_watchdog=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

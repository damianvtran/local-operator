#!/usr/bin/env python3
"""Export the shipped notification sounds from the lossless masters.

WHY THIS EXISTS. ``masters/*.wav`` (24-bit, 44.1 kHz stereo, loudness-matched to
-20 LUFS max momentary by ``master.py``) are the reviewed SOURCE of the ten
built-in notification sounds. They live in git only: at 2.3 MB they would cost
the wheel ~0.8 MiB more than it needs to carry (AGENTS.md "PyPI storage cap").
What ships is derived here, deterministically, into
``local_operator/notifications/sounds/``:

* **FLAC, 16-bit, TPDF-dithered** (``aresample=dither_method=triangular``). FLAC
  is lossless and decodes with ``miniaudio`` (a base dependency) on every OS, so
  playback needs no ``afplay``/``paplay``. 16-bit is transparent here: the
  masters peak at -8 dBFS or lower, so the 16-bit floor sits ~88 dB below the
  signal; the dither keeps the requantisation error noise-like rather than
  correlated distortion on the quiet tails.
* **44.1 kHz, unchanged.** No resampling stage: it would buy nothing, and the OS
  or miniaudio resamples at playback anyway.
* **Mono for eight sounds, stereo for ``soft-pulse`` and ``felt-marimba``.**
  Measured in the design (notification-sounds design §2.3): soft-pulse's side
  energy is -2 dB relative to mid and a mono fold hollows it out (2.3 dB of
  cancellation beyond the expected fold); felt-marimba's -12.7 dB side is
  audible width. The other eight have -19 to -35 dB side energy, so folding them
  is transparent and halves their size.
* **Bit-exact, metadata-free.** ``-fflags +bitexact`` and ``-map_metadata -1``
  keep the encoder's version string out of the file, so re-running this script
  with another ffmpeg build does not churn the committed bytes or the sha256 in
  ``catalog.json`` unless the audio itself changed.

``catalog.json`` records, per sound: ``id``, the ``label`` i18n key (rendered
by the settings surfaces, never stored), ``file``, ``sha256``, ``duration_s``
and ``channels``. ``tests/unit/notifications/test_sound_assets.py`` fails when
the catalog and the files disagree, so the two cannot drift.

Usage (from the repo root; needs a real ffmpeg — on this project's macOS hosts
``/opt/homebrew/bin/ffmpeg``; pass ``--ffmpeg`` to override)::

    python3 assets/notification-sounds/export.py

``master.py`` beside this file is the mastering step that produced the masters
from the generated MP3 takes. It is kept as provenance (it needs numpy and the
original takes, neither of which is in this repository); re-mastering is a
deliberate, reviewed act, while this export can be re-run by anyone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MASTERS = HERE / "masters"
OUT = REPO / "local_operator" / "notifications" / "sounds"

#: Catalog order is the order every picker lists them in. ``soft-pulse`` leads
#: because it is the default (operator's pick, 2026-11-03); the rest follow the
#: review shortlist's order.
SOUNDS: tuple[str, ...] = (
    "soft-pulse",
    "two-note-chime",
    "felt-marimba",
    "hum-swell",
    "task-complete",
    "signal-handshake",
    "tape-warble",
    "bubble-chord",
    "airy-whistle",
    "bubble-pulse",
)

#: The two sounds whose stereo image is part of the sound (see module docstring).
STEREO = frozenset({"soft-pulse", "felt-marimba"})


def _label_key(sound_id: str) -> str:
    """The i18n key of a sound's display label (namespace ``wire.notification_sound``)."""
    return "wire.notification_sound.sound." + sound_id.replace("-", "_")


def _ffmpeg(explicit: str | None) -> str:
    candidate = explicit or "/opt/homebrew/bin/ffmpeg"
    if Path(candidate).is_file():
        return candidate
    found = shutil.which("ffmpeg")
    if not found:
        sys.exit("export.py: no ffmpeg found; pass --ffmpeg PATH")
    return found


def _encode(ffmpeg: str, src: Path, dst: Path, channels: int) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            "-af",
            # Dither at the 24 -> 16 requantisation; the rate is pinned to the
            # source's so the filter never resamples.
            "aresample=44100:osf=s16:dither_method=triangular",
            "-ac",
            str(channels),
            "-c:a",
            "flac",
            "-compression_level",
            "12",
            str(dst),
        ],
        check=True,
    )


def _duration(src: Path) -> float:
    with wave.open(str(src)) as handle:
        return round(handle.getnframes() / handle.getframerate(), 3)


def main(argv: list[str] | None = None) -> int:
    # `__doc__` is None under `python -OO`, which strips docstrings; fall back to
    # a fixed line rather than crash before doing any work.
    summary = __doc__.splitlines()[0] if __doc__ else "Export the shipped notification sounds."
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument("--ffmpeg", default=None)
    args = parser.parse_args(argv)
    ffmpeg = _ffmpeg(args.ffmpeg)
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for sound_id in SOUNDS:
        src = MASTERS / f"{sound_id}.wav"
        dst = OUT / f"{sound_id}.flac"
        channels = 2 if sound_id in STEREO else 1
        _encode(ffmpeg, src, dst, channels)
        entries.append(
            {
                "id": sound_id,
                "label": _label_key(sound_id),
                "file": dst.name,
                "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                "duration_s": _duration(src),
                "channels": channels,
            }
        )
        print(f"{sound_id:17} {channels}ch {dst.stat().st_size:7d} B")
    catalog = {"schema": 1, "default": SOUNDS[0], "sounds": entries}
    (OUT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    total = sum(path.stat().st_size for path in OUT.iterdir())
    print(f"total {total} B ({total / 1048576:.3f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

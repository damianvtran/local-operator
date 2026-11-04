"""The shipped notification sounds: catalog parity, loudness, and the fixed tail.

These pin the properties the design's acceptance names for C1
(notification-sounds design §6): every catalog entry has its file and every
file has its entry (sha256 included), every built-in sits at -20 LUFS max
momentary within ±0.5 LU with peaks <= -1 dBFS, and signal-handshake ends in a
tonal decay, not the broadband "fry" the operator heard in the source take.

The meter is pure Python over miniaudio's decode (no numpy, no ffmpeg), so the
test runs on every CI host. It implements BS.1770's K-weighting (the two
biquads, coefficients derived for 44.1 kHz from the standard's analog
prototype) and the 400 ms momentary window, sliding in 100 ms steps. Each clip
is padded with 0.5 s of silence each side, the padding ``master.py`` measured
with, so a clip shorter than 400 ms still yields a full window.

Mono files are measured as DUAL MONO (decoded to two channels), because that
is how miniaudio's player plays them: each speaker gets the full mono signal.
ebur128 on the raw mono file reads 3 LU lower; that is a metering convention,
not a level change (design §2.3).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from local_operator.notifications import sound

miniaudio = pytest.importorskip("miniaudio")

SR = 44100
CATALOG = json.loads((sound.SOUNDS_DIR / "catalog.json").read_text(encoding="utf-8"))
IDS = [entry["id"] for entry in CATALOG["sounds"]]


def test_the_catalog_and_the_files_are_the_same_set() -> None:
    on_disk = {path.name for path in sound.SOUNDS_DIR.glob("*.flac")}
    listed = {entry["file"] for entry in CATALOG["sounds"]}
    assert on_disk == listed
    for entry in CATALOG["sounds"]:
        digest = hashlib.sha256((sound.SOUNDS_DIR / entry["file"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"], entry["id"]


def test_all_ten_ship_and_soft_pulse_is_the_default() -> None:
    assert len(IDS) == 10
    assert IDS[0] == sound.DEFAULT_SOUND_ID == CATALOG["default"] == "soft-pulse"
    assert sound.sound_ids() == (*IDS, "system")


def test_only_the_two_wide_sounds_are_stereo() -> None:
    stereo = {entry["id"] for entry in CATALOG["sounds"] if entry["channels"] == 2}
    assert stereo == {"soft-pulse", "felt-marimba"}
    for entry in CATALOG["sounds"]:
        info = miniaudio.get_file_info(str(sound.SOUNDS_DIR / entry["file"]))
        assert (info.nchannels, info.sample_rate) == (entry["channels"], SR), entry["id"]


def test_every_label_key_has_a_registry_name() -> None:
    from local_operator.settings_io import _NOTIFICATION_SOUND_LABELS

    assert set(_NOTIFICATION_SOUND_LABELS) == set(IDS)
    for entry in CATALOG["sounds"]:
        assert _NOTIFICATION_SOUND_LABELS[entry["id"]]().code == entry["label"]


def test_the_shipped_sounds_fit_the_wheel_budget() -> None:
    """Design §2.2: the sounds may cost the wheel at most 0.3 MiB."""
    total = sum(path.stat().st_size for path in sound.SOUNDS_DIR.iterdir())
    assert total <= 0.3 * 1024 * 1024


def _biquad(x: list[float], b: tuple[float, ...], a: tuple[float, ...]) -> list[float]:
    y = [0.0] * len(x)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = b
    _, a1, a2 = a
    for i, xi in enumerate(x):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, xi, y1, yi
        y[i] = yi
    return y


def _k_coefficients() -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """BS.1770-4 K-weighting stages, re-derived for 44.1 kHz (libebur128's method)."""
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / SR)
    vh = 10 ** (g / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = (
        ((vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0),
        (1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0),
    )
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / SR)
    a0 = 1 + k / q + k * k
    highpass = ((1.0, -2.0, 1.0), (1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0))
    return shelf, highpass


def _max_momentary_and_peak(path: Path) -> tuple[float, float]:
    decoded = miniaudio.decode_file(
        str(path), output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2, sample_rate=SR
    )
    samples = decoded.samples
    pad = [0.0] * (SR // 2)
    peak = max(abs(v) for v in samples)
    shelf, highpass = _k_coefficients()
    weighted = []
    for channel in (0, 1):
        x = pad + list(samples[channel::2]) + pad
        weighted.append(_biquad(_biquad(x, *shelf), *highpass))
    window, step = int(0.4 * SR), int(0.1 * SR)
    squares = [left * left + right * right for left, right in zip(*weighted)]
    prefix = [0.0]
    for value in squares:
        prefix.append(prefix[-1] + value)
    best = max(
        (prefix[start + window] - prefix[start]) / window
        for start in range(0, len(squares) - window + 1, step)
    )
    return -0.691 + 10 * math.log10(best), 20 * math.log10(peak)


@pytest.mark.parametrize("sound_id", IDS)
def test_every_sound_is_loudness_matched_with_headroom(sound_id: str) -> None:
    loudness, peak = _max_momentary_and_peak(sound.SOUNDS_DIR / f"{sound_id}.flac")
    assert abs(loudness - (-20.0)) <= 0.5, f"{sound_id}: {loudness:.2f} LUFS max momentary"
    assert peak <= -1.0, f"{sound_id}: sample peak {peak:.2f} dBFS"


def test_signal_handshake_ends_in_a_tonal_decay_not_fry() -> None:
    """The source take's tail was broadband noise (ZCR 4-8.7 kHz); the master
    was cut before it. A zero-crossing rate under 2 kHz across the last 60 ms
    is what a decaying tone gives and noise cannot."""
    decoded = miniaudio.decode_file(
        str(sound.SOUNDS_DIR / "signal-handshake.flac"),
        output_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=1,
        sample_rate=SR,
    )
    tail = list(decoded.samples[-int(0.06 * SR) :])
    # Ignore the last few ms of the fade, where the signal sits in the dither.
    tail = tail[: -int(0.005 * SR)]
    crossings = sum(1 for a, b in zip(tail, tail[1:]) if (a < 0) != (b < 0))
    zcr_hz = crossings / 2 / (len(tail) / SR)
    assert zcr_hz < 2000, f"tail zero-crossing rate {zcr_hz:.0f} Hz"

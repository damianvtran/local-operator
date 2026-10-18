"""Waveform peaks: deterministic bars, the degraded states, never a raise.

The recipe is frozen (240 bars, 8 kHz mono, max-abs per bucket, 0..1). The
tests build real WAV bytes in memory — miniaudio decodes them natively — so
the decode path is exercised without any codec fixture. The engine is an
optional extra, so a host without it asserts the DEGRADED contract instead:
``None`` from every entry point, and no exception anywhere.
"""

from __future__ import annotations

import math
import struct

import pytest

from local_operator.audiogen import peaks as peaks_module

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _wav(seconds: float, amplitude: float = 0.5, rate: int = 44100, freq: float = 440.0) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(32767 * amplitude * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(int(rate * seconds))
    )
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(frames))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(frames))
    )
    return header + frames


def _require_engine():
    return pytest.importorskip("miniaudio")


def test_240_bars_bounded_0_to_1_with_a_real_tone():
    _require_engine()
    bars = peaks_module.compute_peaks(_wav(1.0, amplitude=0.5))
    assert bars is not None
    assert len(bars) == peaks_module.PEAK_BARS == 240
    assert all(0.0 <= value <= 1.0 for value in bars)
    # A steady sine's max-abs per bucket is its amplitude (plus float wiggle).
    assert max(bars) == pytest.approx(0.5, abs=0.02)
    assert min(bars) > 0.4


def test_silence_is_zero_bars_not_none():
    # Silence is a REAL waveform: the surface draws a flat line, which is
    # different from "no waveform" (None).
    _require_engine()
    bars = peaks_module.compute_peaks(_wav(0.5, amplitude=0.0))
    assert bars is not None and len(bars) == 240
    assert set(bars) == {0.0}


def test_an_unexpectedly_short_clip_still_yields_240_deterministic_bars():
    _require_engine()
    first = peaks_module.compute_peaks(_wav(1.0))
    second = peaks_module.compute_peaks(_wav(1.0))
    assert first == second
    short = _wav(1 / 8000.0, amplitude=1.0)  # one sample at the decode rate
    assert peaks_module.compute_peaks(short) is not None
    assert len(peaks_module.compute_peaks(short) or []) == 240


def test_undecodable_and_empty_payloads_are_none_never_a_raise():
    _require_engine()
    assert peaks_module.compute_peaks(b"") is None
    assert peaks_module.compute_peaks(b"not audio at all") is None


def test_a_host_without_the_engine_degrades_silently(monkeypatch):
    monkeypatch.setattr(peaks_module, "_load_miniaudio", lambda: None)
    assert peaks_module.compute_peaks(_wav(0.1)) is None


def test_a_decode_failure_degrades_silently(monkeypatch):
    class _Boom:
        class SampleFormat:  # noqa: N801 - mirrors the engine's own shape
            FLOAT32 = 1

        @staticmethod
        def decode(*args, **kwargs):
            raise RuntimeError("decoder exploded")

    monkeypatch.setattr(peaks_module, "_load_miniaudio", lambda: _Boom())
    assert peaks_module.compute_peaks(b"anything") is None


def test_the_payload_shape_is_the_frozen_one():
    payload = peaks_module.peaks_payload([0.25] * 240)
    assert payload["version"] == 1
    assert payload["bars"] == 240
    assert len(payload["peaks"]) == 240

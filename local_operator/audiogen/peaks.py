"""Waveform peaks for generated audio: 240 max-abs bars at 8 kHz mono.

**Why precompute.** A waveform surface wants 240 numbers per clip; a render
loop that decoded the MP3 on every paint would pay a full decode per frame.
The frozen rule is therefore: peaks are computed ONCE, at artifact creation,
and stored beside the bytes as a ``<digest>.peaks.json`` sidecar (see
``session/attachments.py``); nothing decodes in a render loop, and the
waveform route reads the sidecar (with bounded legacy repair in ``recovery``).
The server contract v1 does not emit a
``waveform`` field, so this module is the DEFAULT source — a later server lane
can ship its own peaks and the route prefers the sidecar either way.

**The recipe** (frozen): decode with ``miniaudio`` 1.71 to **8 kHz mono
FLOAT32** (measured 63.7 ms for a 98 s MP3 on this host), split the samples
into 240 even buckets, and take the **max absolute value** per bucket, clamped
to 0..1. 8 kHz mono is deliberate: a waveform bar is a loudness summary, so
the cheapest decode that preserves relative loudness across buckets is the
right one, and the grid is fixed by the contract (``bars: 240``).

**Guarded import, never raises.** ``miniaudio`` is a base runtime dependency,
but a damaged/older install may lack it. Consumers still degrade (no sidecar →
the waveform route answers 404 → the surface draws its no-waveform state).
Import failure, decode failure and empty audio all answer ``None``/silence
rather than raising: peaks are presentation, and no render surface may take a
turn down.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["PEAK_BARS", "PEAKS_VERSION", "compute_duration", "compute_peaks", "peaks_payload"]

#: The frozen bar count (server-contract-adjacent vocabulary: ``bars: 240``).
PEAK_BARS = 240
#: The sidecar/payload schema version (``version: 1``).
PEAKS_VERSION = 1
#: The decode grid: 8 kHz mono. Deliberately low — see the module docstring.
PEAK_DECODE_SAMPLE_RATE_HZ = 8_000


def _load_miniaudio() -> Any | None:
    """The optional engine, or ``None``.

    ``importlib`` rather than a top-level ``import miniaudio``: the dependency
    is a native decoder whose import can fail even in a complete install.
    Keep that failure a supported degraded state, not a failed generation.
    """
    try:
        return importlib.import_module("miniaudio")
    except Exception:  # noqa: BLE001 - any import failure is the degraded state
        logger.debug("miniaudio unavailable; peaks will not be computed", exc_info=True)
        return None


def _decode(raw: bytes) -> Any | None:
    """The one guarded decode shared by peaks and duration.

    Same engine, same grid, same degraded state: ``None`` means this host
    cannot read these bytes, which both callers treat as "no derived fact"
    rather than as an error. Kept in one place so the two derivations cannot
    drift onto different decode settings — a duration taken at another sample
    rate would still be arithmetically right, but a peak grid taken at another
    rate would not, and one shared call is how that stays impossible.
    """
    if not raw:
        return None
    miniaudio = _load_miniaudio()
    if miniaudio is None:
        return None
    try:
        return miniaudio.decode(
            raw,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=PEAK_DECODE_SAMPLE_RATE_HZ,
        )
    except Exception:  # noqa: BLE001 - undecodable input is ordinary here
        logger.debug("peaks: decode failed; reporting no waveform", exc_info=True)
        return None


def compute_duration(raw: bytes) -> float | None:
    """The clip's length in seconds, read from the bytes, or ``None``.

    **THE DURATION IS A PROPERTY OF THE BYTES, NOT OF THE PROVIDER'S RESPONSE.**
    A provider that states a duration is offering a cross-check; a provider that
    does not (ElevenLabs' text-to-sound answers with the requested FORMAT and
    nothing else) must not leave the artifact without one. The consequence of
    leaving it out is user-visible rather than cosmetic: the sound card needs a
    length to draw a waveform and a clock, so a null duration renders a record
    with NO PLAYER — a generated sound the user cannot play, with no error.

    Derived from the same 8 kHz mono decode as the peaks: ``frames / rate`` is
    exact for any format the decoder accepts, including MP3, whose container
    padding the decoder accounts for. ``None`` keeps the old degraded state
    (no engine, undecodable bytes) and callers keep whatever they were given.
    """
    decoded = _decode(raw)
    if decoded is None:
        return None
    samples = getattr(decoded, "samples", None)
    rate = getattr(decoded, "sample_rate", None)
    if samples is None or not rate:
        return None
    return len(samples) / float(rate)


def compute_peaks(raw: bytes) -> list[float] | None:
    """240 max-abs bars for ``raw`` audio bytes, or ``None`` when uncomputable.

    ``None`` means "this host cannot summarise these bytes" — no miniaudio, an
    undecodable payload, an empty one. Consumers treat it as "no waveform",
    never as an error. A successfully decoded SILENT clip returns 240 zeros,
    which is different from ``None``: silence is a real waveform.
    """
    decoded = _decode(raw)
    if decoded is None:
        return None
    samples = getattr(decoded, "samples", None)
    if samples is None:
        return None
    total = len(samples)
    bars = [0.0] * PEAK_BARS
    if total == 0:
        return bars
    # One pass, index-mapped buckets: bucket i owns sample indices whose
    # scaled position falls in [i, i+1) — even for clips shorter than 240
    # samples, and without building a per-bucket slice list.
    for index, value in enumerate(samples):
        bucket = index * PEAK_BARS // total
        try:
            magnitude = abs(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if magnitude > bars[bucket]:
            bars[bucket] = magnitude
    return [min(1.0, max(0.0, round(value, 4))) for value in bars]


def peaks_payload(peaks: list[float]) -> dict[str, Any]:
    """The one wire/sidecar payload for a peaks list.

    ``{version, bars, peaks}`` — the frozen shape the waveform route returns
    and the sidecar stores. Built here so the two spellings cannot drift.
    """
    return {"version": PEAKS_VERSION, "bars": PEAK_BARS, "peaks": [float(v) for v in peaks]}

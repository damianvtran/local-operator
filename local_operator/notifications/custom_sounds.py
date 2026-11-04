"""The user's own notification sounds: the store, the import pipeline, the render cache.

WHAT THIS IS FOR. The operator asked to "add and set their own custom sound
effects (allow multiple)", including a clip an agent just made with
``generate_sound_effect``. This module is the one place that turns arbitrary
audio bytes into a sound ``notifications.sound.id`` can name. Its three callers
are ``lop sound add|rename|remove`` (``sound_cli``), the desktop routes
(``server/routes/notification_sounds.py``) and the player's resolver
(``notifications.sound.resolve``). Design: notification-sounds design §3.

THE STORE (design §3.1). ``<config>/sounds/custom/<id>.wav`` plus
``<config>/sounds/custom/index.json``. ``config_dir()`` honours
``LOCAL_OPERATOR_CONFIG_DIR``, so a rig or a test isolates the store by setting
it. The index is the authority: a ``.wav`` with no index row is invisible, and
an index row whose file is missing resolves to nothing (the resolver then falls
back to the default sound, never to silence).

* ids are ``custom-<slug>-<6 hex>`` and must match :data:`ID_RE`. The prefix
  keeps them disjoint from built-in ids by SHAPE, so no built-in added later can
  collide with a stored id.
* caps: :data:`MAX_SOUNDS` sounds, :data:`MAX_INPUT_BYTES` of input,
  :data:`MAX_DURATION_S` of sound after silence is trimmed.
* stored as 16-bit WAV, not FLAC (design §3.3 step 6): miniaudio has no FLAC
  encoder and a second dependency is not worth it; a 10 s stereo 48 kHz WAV is
  1.9 MB, so the caps still bound the disk.

OVER-LENGTH CLIPS ARE REFUSED, NOT TRIMMED (decision on settings spec C10). A
3-minute song silently cut to its first 10 s becomes a 10 s notification, which
is exactly the intrusiveness the operator asked to avoid; a refusal tells the
user to pick a short clip. Leading and trailing SILENCE is still trimmed first,
because that is padding, not content, and :data:`DURATION_SLOP_S` absorbs
decoder rounding.

IMPORT PIPELINE (design §3.3). Magic sniff, then a FULL decode with miniaudio
as the real validator, then: trim silence below -50 dBFS, short fades so a
trimmed edge cannot click, loudness-match to -20 LUFS max momentary with the
sample peak capped at -1 dBFS (the built-ins' target, so the volume slider
means the same thing for every sound), keep at most 48 kHz, keep stereo only
when the side channel carries something (> -15 dB under mid). ``keep_loudness``
skips the loudness match for someone who mastered their own file.

SECURITY (design §3.5).

* A filesystem path is only ever ``root / f"{id}.wav"``, the id regex-checked
  AND the resolved path checked to sit inside the resolved root. The user's
  filename is stored as ``source_name`` text and never touches the filesystem.
* Symlinks are refused: a store root, a sound file or the index that is a
  symlink is treated as absent/unusable, so a planted link cannot make the
  audio route or the player read outside the store.
* Writes go to a temp file in the same directory and are renamed after
  validation, so a crash leaves no partial sound.
* Index mutations and the "removed the selected sound" config reset run under
  ``config_write_lock``, the config dir's own writer lock. It is reentrant on
  one thread, so the reset (which takes the same lock inside ``settings_io``)
  happens inside the same critical section as the removal.

STDLIB-ONLY AT IMPORT, like ``notifications.sound``: the resolver imports this
on every notification. miniaudio and the settings registry load inside the
functions that need them.
"""

from __future__ import annotations

import array
import dataclasses
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: At most this many custom sounds. A picker of 50 rows is already long; the
#: cap bounds the disk to about 100 MB in the worst case (50 x 1.9 MB).
MAX_SOUNDS = 50
#: Input ceiling in bytes, checked before any decode (and while streaming an
#: upload). 10 MB holds a 10 s 24-bit 96 kHz stereo WAV with room to spare.
MAX_INPUT_BYTES = 10 * 1024 * 1024
#: Longest sound accepted, measured after leading/trailing silence is trimmed.
MAX_DURATION_S = 10.0
#: Decoder rounding allowance on top of :data:`MAX_DURATION_S` (an MP3's
#: encoder delay and padding add a few tens of ms to an exactly-10 s clip).
DURATION_SLOP_S = 0.05
#: The most audio the import ever decodes, BEFORE silence is trimmed: a header
#: claiming more is refused without decoding, and the streamed decode stops
#: (and refuses) once it has produced more, whatever the header said. 30 s
#: leaves room for generous silence padding around a real 10 s clip, and bounds
#: the decode buffer at ~11.5 MB of float32 (48 kHz stereo).
PRECHECK_DURATION_S = 30.0
#: Frames per decode chunk. Small enough that the overshoot past the frame cap
#: is negligible, large enough that the Python loop costs nothing measurable.
_STREAM_CHUNK_FRAMES = 4096
#: Longest display name, matching the desktop's inline-rename validation.
MAX_NAME_CHARS = 40

ID_RE = re.compile(r"^custom-[a-z0-9-]{1,40}$")
_SLUG_MAX = 32

#: Loudness target and peak ceiling: the built-in masters' own (design §2.3).
TARGET_LUFS = -20.0
PEAK_CEILING_DBFS = -1.0
SILENCE_DBFS = -50.0
#: Keep a second channel only when side energy is within 15 dB of mid.
STEREO_SIDE_DB = -15.0
MAX_SAMPLE_RATE = 48000
_FADE_IN_S = 0.005
_FADE_OUT_S = 0.010
#: A first sample above this is "hot": starting playback there would click.
_HOT_DBFS = -40.0

#: Rendered-at-a-volume WAVs kept for the desktop's audio route. The key
#: includes the content sha, so a rename never serves stale bytes and a
#: re-import under the same id cannot either. Pruned oldest-first past this
#: many files, because 60 sounds x 101 volumes would otherwise be gigabytes.
RENDER_CACHE_MAX_FILES = 64

INDEX_FILE = "index.json"
_INDEX_VERSION = 1

#: Container magic that names a format miniaudio cannot decode. Refused with
#: ``unsupported_format`` and the format, so the user is told to convert it
#: rather than that their file is broken (design §3.2: measured, miniaudio
#: decodes neither Opus nor AAC).
_UNSUPPORTED_CODECS = frozenset({"opus", "m4a", "aac"})


class SoundRejected(Exception):
    """An add, rename or remove that was refused, with a machine ``code``.

    ``code`` is the wire vocabulary the desktop switches on (design §3.4 and the
    settings spec §5.2): ``too_large``, ``unsupported_format``, ``too_long``,
    ``undecodable``, ``limit_reached``, ``not_found``, ``builtin_readonly``,
    ``invalid_name``, ``store_failed``. ``message`` is the keyed ``Msg`` whose
    ``text`` the CLI prints and the route sends; ``status`` is the HTTP status.
    """

    def __init__(
        self, code: str, status: int, message: Any, *, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.message = message
        #: Machine params the copy does not interpolate but a client switches
        #: on (``format`` for ``unsupported_format``: the settings spec picks
        #: its Opus/M4A sentence from it, and never parses ``text``).
        self.extra = dict(extra or {})

    def envelope(self) -> dict[str, Any]:
        body = self.message.envelope(code=self.code)
        body["params"] = {**body.get("params", {}), **self.extra}
        return body

    @property
    def text(self) -> str:
        """The refusal in the process's ACTIVE locale (what the CLI prints).

        ``envelope()`` stays en-by-default like every route envelope (the
        desktop keys on ``code``); a terminal line has no client to localize
        it, so it renders here (QA round 2, Q6).
        """
        from local_operator.i18n.text import active_locale

        return str(self.message.envelope(locale=active_locale())["text"])


@dataclasses.dataclass(frozen=True)
class CustomSound:
    """One index row."""

    id: str
    name: str
    source_name: str
    sha256: str
    duration_s: float
    created_at: str
    normalized: bool
    channels: int = 1
    sample_rate: int = 44100

    def to_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _keys() -> Any:
    from local_operator.i18n.keys import wire_notification_sound as _ns

    return _ns


# -- paths ------------------------------------------------------------------


def sounds_root() -> Path:
    from local_operator.paths import config_dir

    return config_dir() / "sounds"


def store_root() -> Path:
    return sounds_root() / "custom"


def render_root() -> Path:
    return sounds_root() / "rendered"


def is_custom_id(sound_id: object) -> bool:
    return isinstance(sound_id, str) and ID_RE.fullmatch(sound_id) is not None


def _contained(root: Path, name: str) -> Path | None:
    """``root / name`` if it resolves inside ``root`` and nothing is a symlink.

    The regex already forbids ``/`` and ``..`` in an id; the resolve check is
    the second wall, so a future change to the id shape cannot open a hole on
    its own. ``root`` itself being a symlink is refused too: a store pointed
    elsewhere is a store this module did not create.
    """
    if root.is_symlink():
        return None
    candidate = root / name
    if candidate.is_symlink():
        return None
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved.parent != resolved_root:
        return None
    return resolved


def sound_path(sound_id: str) -> Path | None:
    """The stored file for ``sound_id``, or ``None`` if the id is not a safe custom id."""
    if not is_custom_id(sound_id):
        return None
    return _contained(store_root(), f"{sound_id}.wav")


# -- the index ----------------------------------------------------------------


def _index_path() -> Path:
    return store_root() / INDEX_FILE


def list_sounds() -> tuple[CustomSound, ...]:
    """Every indexed custom sound, oldest first. Never raises.

    A missing, unreadable or malformed index is an empty store: this is read on
    every notification and on every settings page render, and neither may fail
    because a hand edit broke a file. A row that fails validation is skipped,
    not repaired, so a bad row cannot smuggle an unsafe id into a path.
    """
    path = _index_path()
    if path.is_symlink() or store_root().is_symlink():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    rows = raw.get("sounds") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return ()
    sounds: list[CustomSound] = []
    seen: set[str] = set()
    for row in rows:
        try:
            sound = CustomSound(
                id=str(row["id"]),
                name=str(row["name"]),
                source_name=str(row.get("source_name", "")),
                sha256=str(row["sha256"]),
                duration_s=float(row["duration_s"]),
                created_at=str(row.get("created_at", "")),
                normalized=bool(row.get("normalized", True)),
                channels=int(row.get("channels", 1)),
                sample_rate=int(row.get("sample_rate", 44100)),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if not is_custom_id(sound.id) or sound.id in seen:
            continue
        seen.add(sound.id)
        sounds.append(sound)
    return tuple(sounds)


def get(sound_id: str) -> CustomSound | None:
    if not is_custom_id(sound_id):
        return None
    for sound in list_sounds():
        if sound.id == sound_id:
            return sound
    return None


def resolve_file(sound_id: str) -> tuple[CustomSound, Path] | None:
    """``(row, path)`` for a custom sound that can actually be played, else ``None``."""
    sound = get(sound_id)
    if sound is None:
        return None
    path = sound_path(sound.id)
    if path is None or not path.is_file():
        return None
    return sound, path


def _write_index(sounds: list[CustomSound]) -> None:
    root = store_root()
    payload = {"version": _INDEX_VERSION, "sounds": [s.to_row() for s in sounds]}
    tmp = root / f".{INDEX_FILE}.{secrets.token_hex(4)}.tmp"
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, root / INDEX_FILE)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _locked() -> Iterator[Path]:
    """The store root, created, under the config dir's writer lock."""
    from local_operator.config_writer import config_write_lock
    from local_operator.paths import config_dir

    root = store_root()
    with config_write_lock(config_dir()):
        for directory in (sounds_root(), root):
            if directory.is_symlink():
                raise _store_failed()
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if _index_path().is_symlink():
            raise _store_failed()
        yield root


def _store_failed() -> SoundRejected:
    from local_operator.session.store_failures import display_root

    return SoundRejected(
        "store_failed", 503, _keys().error_store_failed(where=display_root(store_root()))
    )


# -- names and ids ------------------------------------------------------------


def clean_name(name: str) -> str:
    """A display name, validated. Raises :class:`SoundRejected` (``invalid_name``)."""
    # Control characters are STRIPPED (QA round 1, Q2: deferred, not refused).
    # Refusing needs a new keyed sentence, which would change the audited
    # `wire.notification_sound` source and un-ship every locale until a delta
    # audit; stripping keeps the visible name the user typed.
    text = "".join(ch for ch in str(name) if ch.isprintable() or ch in " \t\n\r")
    cleaned = " ".join(text.split())
    if not cleaned:
        raise SoundRejected("invalid_name", 422, _keys().error_name_empty())
    if len(cleaned) > MAX_NAME_CHARS:
        raise SoundRejected("invalid_name", 422, _keys().error_name_long(limit=str(MAX_NAME_CHARS)))
    return cleaned


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:_SLUG_MAX].strip("-")
    return slug or "sound"


def new_id(name: str, taken: set[str]) -> str:
    slug = _slug(name)
    while True:
        candidate = f"custom-{slug}-{secrets.token_hex(3)}"
        if candidate not in taken and ID_RE.fullmatch(candidate):
            return candidate


def default_name(source_name: str) -> str:
    """A name from the source's file stem, clipped to the name limit."""
    stem = Path(source_name).stem if source_name else ""
    stem = " ".join(re.sub(r"[_]+", " ", stem).split())
    return stem[:MAX_NAME_CHARS].strip() or "Custom sound"


# -- sniff and decode -----------------------------------------------------------


def sniff(raw: bytes) -> str | None:
    """The container a file's magic bytes name, or ``None`` when nothing matches.

    Returns one of ``wav flac mp3 vorbis`` (decodable) or ``opus m4a aac ogg
    webm`` (refused with ``unsupported_format``). ``None`` means "not an audio
    file this recognises", which is reported as ``undecodable``: a text file
    renamed ``.wav`` is not an unsupported FORMAT, it is not audio at all.
    """
    head = raw[:4096]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        if b"OpusHead" in head:
            return "opus"
        if b"\x01vorbis" in head:
            return "vorbis"
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[:3] == b"ID3":
        return "mp3"
    if len(head) >= 2 and head[0] == 0xFF:
        # ADTS AAC is a 12-bit sync with layer bits 00; MPEG audio layers 1-3
        # share the 11-bit sync but never carry layer 00.
        if head[1] & 0xF6 == 0xF0:
            return "aac"
        if head[1] & 0xE0 == 0xE0 and (head[1] >> 1) & 0x03 != 0:
            return "mp3"
    return None


@dataclasses.dataclass
class Prepared:
    """A decoded, processed clip ready to write."""

    samples: "array.array[int]"
    channels: int
    sample_rate: int
    duration_s: float
    normalized: bool


def _db(value: float) -> float:
    return 20 * math.log10(value) if value > 0 else -math.inf


def _biquad(
    x: "array.array[float] | list[float]", b: tuple[float, float, float], a: tuple[float, float]
) -> list[float]:
    y = [0.0] * len(x)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = b
    a1, a2 = a
    for i, xi in enumerate(x):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, xi, y1, yi
        y[i] = yi
    return y


def _k_weighting(sample_rate: int) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    """BS.1770-4 K-weighting stages for ``sample_rate`` (libebur128's derivation).

    The same filter ``tests/unit/notifications/test_sound_assets.py`` uses to
    pin the built-ins at -20 LUFS, so a custom sound and a built-in at one
    slider position sit at one loudness.
    """
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sample_rate)
    vh = 10 ** (g / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = (
        ((vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0),
        (2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0),
    )
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sample_rate)
    a0 = 1 + k / q + k * k
    highpass = ((1.0, -2.0, 1.0), (2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0))
    return shelf, highpass


def max_momentary_lufs(channels: list[list[float]], sample_rate: int) -> float:
    """Max momentary loudness (400 ms windows, 100 ms hop) of float channels."""
    shelf, highpass = _k_weighting(sample_rate)
    pad = [0.0] * (sample_rate // 2)
    squares: list[float] | None = None
    for channel in channels:
        weighted = _biquad(_biquad(pad + channel + pad, *shelf), *highpass)
        if squares is None:
            squares = [v * v for v in weighted]
        else:
            squares = [s + v * v for s, v in zip(squares, weighted)]
    assert squares is not None
    window, step = int(0.4 * sample_rate), int(0.1 * sample_rate)
    prefix = [0.0]
    for value in squares:
        prefix.append(prefix[-1] + value)
    best = max(
        (
            (prefix[start + window] - prefix[start]) / window
            for start in range(0, max(1, len(squares) - window + 1), step)
        ),
        default=0.0,
    )
    return -0.691 + 10 * math.log10(best) if best > 0 else -math.inf


def _too_long(file_label: str, seconds: float) -> SoundRejected:
    return SoundRejected(
        "too_long",
        422,
        _keys().error_too_long(
            file=file_label, seconds=f"{seconds:.1f}", limit=f"{MAX_DURATION_S:g}"
        ),
    )


def prepare(raw: bytes, *, file_label: str, keep_loudness: bool = False) -> Prepared:
    """Validate and process ``raw``; raises :class:`SoundRejected` on refusal.

    ``file_label`` is the user's name for the file, used ONLY in refusal text.
    """
    _ns = _keys()
    if len(raw) > MAX_INPUT_BYTES:
        raise SoundRejected(
            "too_large",
            413,
            _ns.error_too_large(file=file_label, limit_mb=str(MAX_INPUT_BYTES // (1024 * 1024))),
        )
    kind = sniff(raw)
    if kind is None:
        raise SoundRejected("undecodable", 422, _ns.error_undecodable(file=file_label))
    if kind in _UNSUPPORTED_CODECS:
        raise SoundRejected(
            "unsupported_format",
            415,
            _ns.error_unsupported_codec(file=file_label),
            extra={"format": kind},
        )
    if kind not in ("wav", "flac", "mp3", "vorbis"):
        raise SoundRejected(
            "unsupported_format",
            415,
            _ns.error_unsupported_format(file=file_label),
            extra={"format": kind},
        )

    import miniaudio

    info_reader = {
        "wav": miniaudio.wav_get_info,
        "flac": miniaudio.flac_get_info,
        "mp3": miniaudio.mp3_get_info,
        "vorbis": miniaudio.vorbis_get_info,
    }[kind]
    # The header's duration is a HINT, never a bound (agent review round 1,
    # R1): FLAC STREAMINFO may say "length unknown" (0) or simply lie, and a
    # 213 KB FLAC of 20 minutes of near-silence decoded whole costs 911 MB of
    # float32 before any check runs. So the decode is a STREAM that stops,
    # and refuses, once it has produced more than PRECHECK_DURATION_S of
    # audio: at most ~11.5 MB of float32 (30 s of 48 kHz stereo) is ever held,
    # whatever the container claims. The header still answers the common case
    # cheaply (a truthful 3-minute song is refused before decoding a frame).
    try:
        info = info_reader(raw)
        source_channels = max(1, min(2, int(info.nchannels)))
        sample_rate = min(int(info.sample_rate), MAX_SAMPLE_RATE)
        if info.duration > PRECHECK_DURATION_S:
            raise _too_long(file_label, info.duration)
        if sample_rate <= 0:
            raise SoundRejected("undecodable", 422, _ns.error_undecodable(file=file_label))
        frame_cap = int(PRECHECK_DURATION_S * sample_rate)
        interleaved = array.array("f")
        for chunk in miniaudio.stream_memory(
            raw,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=source_channels,
            sample_rate=sample_rate,
            frames_to_read=_STREAM_CHUNK_FRAMES,
        ):
            interleaved.extend(chunk)
            if len(interleaved) // source_channels > frame_cap:
                # Honest about what is known: more than the cap was decoded,
                # and the header (when it said anything) is the best total.
                raise _too_long(file_label, max(info.duration, PRECHECK_DURATION_S))
    except miniaudio.MiniaudioError:
        raise SoundRejected("undecodable", 422, _ns.error_undecodable(file=file_label)) from None
    frames = len(interleaved) // source_channels
    if frames == 0:
        raise SoundRejected("undecodable", 422, _ns.error_undecodable(file=file_label))
    channels = [list(interleaved[c::source_channels]) for c in range(source_channels)]

    # 1. Trim leading and trailing silence (padding, not content).
    threshold = 10 ** (SILENCE_DBFS / 20)
    first = next((i for i in range(frames) if any(abs(ch[i]) > threshold for ch in channels)), None)
    if first is None:
        # A file of pure silence decodes, but there is nothing to play: the
        # same code as a broken file (the desktop has one row for both), with
        # its own sentence so the CLI user learns why.
        raise SoundRejected("undecodable", 422, _ns.error_silent(file=file_label))
    last = next(
        i for i in range(frames - 1, -1, -1) if any(abs(ch[i]) > threshold for ch in channels)
    )
    channels = [ch[first : last + 1] for ch in channels]
    frames = last + 1 - first
    duration = frames / sample_rate

    # 2. Refuse, never truncate, an over-length clip (settings spec C10).
    if duration > MAX_DURATION_S + DURATION_SLOP_S:
        raise _too_long(file_label, duration)

    # 3. Fades: a trimmed edge sits exactly where the signal crossed -50 dBFS,
    # and a hot first sample would click as the device starts.
    hot = 10 ** (_HOT_DBFS / 20)
    fade_in = min(frames, int(_FADE_IN_S * sample_rate))
    if fade_in and any(abs(ch[0]) > hot for ch in channels):
        for ch in channels:
            for i in range(fade_in):
                ch[i] *= i / fade_in
    fade_out = min(frames, int(_FADE_OUT_S * sample_rate))
    for ch in channels:
        for i in range(fade_out):
            ch[frames - 1 - i] *= i / fade_out

    # 4. Mono unless the side channel is worth its bytes.
    if len(channels) == 2:
        left, right = channels
        mid = sum(((a + b) * 0.5) ** 2 for a, b in zip(left, right))
        side = sum(((a - b) * 0.5) ** 2 for a, b in zip(left, right))
        if mid <= 0 or 10 * math.log10(max(side, 1e-30) / mid) < STEREO_SIDE_DB:
            channels = [[(a + b) * 0.5 for a, b in zip(left, right)]]

    # 5. Loudness-match to the built-ins' target, peak-capped.
    normalized = not keep_loudness
    peak = max(max((abs(v) for v in ch), default=0.0) for ch in channels)
    if normalized:
        loudness = max_momentary_lufs(channels, sample_rate)
        gain = 10 ** ((TARGET_LUFS - loudness) / 20) if math.isfinite(loudness) else 1.0
        ceiling = 10 ** (PEAK_CEILING_DBFS / 20)
        if peak * gain > ceiling:
            gain = ceiling / peak
    else:
        # Untouched loudness, but never past full scale.
        gain = 1.0 / peak if peak > 1.0 else 1.0

    out = array.array("h", bytes(2 * frames * len(channels)))
    width = len(channels)
    for c, ch in enumerate(channels):
        for i, value in enumerate(ch):
            out[i * width + c] = max(-32768, min(32767, int(round(value * gain * 32767))))
    return Prepared(out, width, sample_rate, round(duration, 3), normalized)


# -- mutations ------------------------------------------------------------------


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def add(
    raw: bytes,
    *,
    source_name: str,
    name: str | None = None,
    keep_loudness: bool = False,
) -> CustomSound:
    """Import ``raw`` as a new custom sound and return its index row.

    ``source_name`` is the user's filename (or an attachment digest), kept as
    text for display and refusal copy; it never becomes a path.
    """
    label = Path(source_name).name or source_name or "sound"
    display = clean_name(name) if name is not None else default_name(label)
    # Cheap refusal first: decoding 10 MB to learn the store is full is waste.
    if len(list_sounds()) >= MAX_SOUNDS:
        raise SoundRejected(
            "limit_reached", 422, _keys().error_limit_reached(limit=str(MAX_SOUNDS))
        )
    prepared = prepare(raw, file_label=label, keep_loudness=keep_loudness)

    import miniaudio

    with _locked() as root:
        sounds = list(list_sounds())
        if len(sounds) >= MAX_SOUNDS:
            raise SoundRejected(
                "limit_reached", 422, _keys().error_limit_reached(limit=str(MAX_SOUNDS))
            )
        sound_id = new_id(display, {s.id for s in sounds})
        target = _contained(root, f"{sound_id}.wav")
        if target is None:
            raise _store_failed()
        tmp = root / f".{sound_id}.{secrets.token_hex(4)}.tmp.wav"
        try:
            miniaudio.wav_write_file(
                str(tmp),
                miniaudio.DecodedSoundFile(
                    display,
                    prepared.channels,
                    prepared.sample_rate,
                    miniaudio.SampleFormat.SIGNED16,
                    prepared.samples,
                ),
            )
            sha = _sha256(tmp)
            os.replace(tmp, target)
        except OSError:
            raise _store_failed() from None
        finally:
            tmp.unlink(missing_ok=True)
        row = CustomSound(
            id=sound_id,
            name=display,
            source_name=label,
            sha256=sha,
            duration_s=prepared.duration_s,
            created_at=_now(),
            normalized=prepared.normalized,
            channels=prepared.channels,
            sample_rate=prepared.sample_rate,
        )
        try:
            _write_index([*sounds, row])
        except OSError:
            target.unlink(missing_ok=True)
            raise _store_failed() from None
    return row


def _require_custom(sound_id: str) -> None:
    from local_operator.notifications.sound import SYSTEM_SOUND_ID, builtin

    if sound_id == SYSTEM_SOUND_ID or builtin(sound_id) is not None:
        raise SoundRejected("builtin_readonly", 409, _keys().error_builtin_readonly(id=sound_id))


def _not_found(sound_id: str) -> SoundRejected:
    # The id is echoed back as text only; a hostile id (``../x``) is display
    # copy here and never reaches a path.
    return SoundRejected("not_found", 404, _keys().error_not_found(id=sound_id[:80]))


def rename(sound_id: str, name: str) -> CustomSound:
    _require_custom(sound_id)
    display = clean_name(name)
    if not is_custom_id(sound_id):
        raise _not_found(sound_id)
    with _locked():
        sounds = list(list_sounds())
        for index, sound in enumerate(sounds):
            if sound.id == sound_id:
                updated = dataclasses.replace(sound, name=display)
                sounds[index] = updated
                try:
                    _write_index(sounds)
                except OSError:
                    raise _store_failed() from None
                return updated
    raise _not_found(sound_id)


@dataclasses.dataclass(frozen=True)
class Removal:
    sound: CustomSound
    #: True when the removed sound was the selected one and
    #: ``notifications.sound.id`` was reset to the default in the same lock.
    selection_reset: bool

    def reset_message(self) -> Any:
        """The keyed "selected again" message, naming the default's display name."""
        from local_operator.notifications.sound import DEFAULT_SOUND_ID

        default = DEFAULT_SOUND_ID
        try:
            from local_operator.settings_io import BY_KEY

            for choice in BY_KEY["notifications.sound.id"].resolved_choices:
                if choice.value == DEFAULT_SOUND_ID:
                    default = str(choice.description) or DEFAULT_SOUND_ID
        except Exception:  # noqa: BLE001 — the id is an honest fallback name
            pass
        return _keys().removed_selection_reset(default=default)

    def reset_text(self) -> str:
        """The "selected again" line in the active locale (CLI output)."""
        from local_operator.i18n.text import active_locale

        return str(self.reset_message().envelope(locale=active_locale())["text"])


class _ResetAuditFilter(logging.Filter):
    """Drops ConfigManager's "does not carry notifications.sound.id through" record.

    That WARNING is ``ConfigManager``'s audit line for every declared drop, and
    its reason names ``/settings`` (agent review round 1, R4). For this caller it
    is noise on the user's terminal, under the keyed "Soft pulse is selected
    again" line that already says what happened. Scoped to the CURRENT THREAD
    and this one key, so a concurrent writer's record in a server is untouched.
    """

    def __init__(self) -> None:
        super().__init__()
        self._thread = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.thread != self._thread:
            return True
        return "notifications.sound.id" not in str(record.args)


@contextmanager
def _quiet_reset_audit() -> Iterator[None]:
    target = logging.getLogger("local_operator.config")
    quiet = _ResetAuditFilter()
    target.addFilter(quiet)
    try:
        yield
    finally:
        target.removeFilter(quiet)


def remove(sound_id: str) -> Removal:
    """Delete one custom sound; reset the selection to the default if it was selected.

    The reset happens in the same critical section as the deletion, so no
    notification can resolve an id whose file has just gone (it would fall
    back to the default anyway, but the config would keep naming a ghost).
    """
    _require_custom(sound_id)
    if not is_custom_id(sound_id):
        raise _not_found(sound_id)
    from local_operator.notifications import sound as _sound

    with _locked() as root:
        sounds = list(list_sounds())
        match = next((s for s in sounds if s.id == sound_id), None)
        if match is None:
            raise _not_found(sound_id)
        try:
            _write_index([s for s in sounds if s.id != sound_id])
        except OSError:
            raise _store_failed() from None
        target = _contained(root, f"{sound_id}.wav")
        if target is not None:
            target.unlink(missing_ok=True)
        else:
            # A planted symlink in the sound's slot (QA round 1, Q4): unlink the
            # LINK itself, never its target. `Path.unlink` does not follow it,
            # and the id regex already pins the name inside `root`.
            planted = root / f"{sound_id}.wav"
            if planted.is_symlink():
                planted.unlink(missing_ok=True)
        _drop_renders(sound_id)
        selected = _sound._walk(_sound._config_values(), "notifications", "sound", "id")
        reset = selected == sound_id
        if reset:
            from local_operator import settings_io
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            with _quiet_reset_audit():
                settings_io.reset_setting(
                    ConfigManager(config_dir()), settings_io.BY_KEY["notifications.sound.id"]
                )
    return Removal(match, reset)


# -- render cache (the desktop's audio route) -------------------------------------


def _drop_renders(sound_id: str) -> None:
    root = render_root()
    if root.is_symlink() or not root.is_dir():
        return
    for path in root.glob(f"{sound_id}-*.wav"):
        if not path.is_symlink():
            path.unlink(missing_ok=True)


def _prune_renders(root: Path) -> None:
    files = [p for p in root.glob("*.wav") if p.is_file() and not p.is_symlink()]
    if len(files) <= RENDER_CACHE_MAX_FILES:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    for path in files[: len(files) - RENDER_CACHE_MAX_FILES]:
        path.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class Rendered:
    path: Path
    etag: str
    #: The file's bytes, read while this call still owns it: the route serves
    #: these, so a later prune of ``path`` cannot fail an in-flight response.
    data: bytes


def render(sound_id: str, volume: int) -> Rendered | None:
    """``sound_id`` as a 16-bit WAV at the gain for ``volume``; ``None`` if unknown.

    Serves only ids the catalog or the index name: the id is looked up, never
    joined to a path from the request. The gain is the notification's own
    ``(v/100)**2`` curve, applied to the samples, so a desktop preview at a
    volume sounds like the notification at that volume.
    """
    from local_operator.notifications import play
    from local_operator.notifications import sound as _sound

    entry = _sound.playable(sound_id)
    if entry is None:
        return None
    _, source, sha = entry
    volume = max(0, min(100, int(volume)))
    root = render_root()
    if sounds_root().is_symlink() or root.is_symlink():
        return None
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = f"{sound_id}-{sha[:8]}-{volume}.wav"
    target = _contained(root, name)
    if target is None:
        return None
    etag = f'"{sha[:16]}-{volume}"'
    if target.is_file():
        try:
            return Rendered(target, etag, target.read_bytes())
        except FileNotFoundError:
            pass  # pruned between the check and the read: render it again

    import miniaudio

    info = miniaudio.get_file_info(str(source))
    decoded = miniaudio.decode_file(
        str(source),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=info.nchannels,
        sample_rate=info.sample_rate,
    )
    scaled = play._scaled(decoded.samples, _sound.gain(volume))
    tmp = root / f".{name}.{secrets.token_hex(4)}.tmp"
    try:
        miniaudio.wav_write_file(
            str(tmp),
            miniaudio.DecodedSoundFile(
                sound_id, info.nchannels, info.sample_rate, miniaudio.SampleFormat.SIGNED16, scaled
            ),
        )
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    data = target.read_bytes()
    _prune_renders(root)
    return Rendered(target, etag, data)

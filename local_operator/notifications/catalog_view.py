"""One reading of "which notification sounds exist", shared by every surface that lists them.

WHY THIS MODULE EXISTS (notification-sounds addendum 2 §2.3). Two routes list
the sounds: the desktop's ``GET /v1/notification-sounds`` (C2,
``server/routes/notification_sounds.py``) and the paired phone's
``GET /api/notification-sounds`` on the mobile daemon (C5). They answer
different callers with different extra fields, but the ROWS — which ids exist,
in which order, under which localized label — must be one answer, or the phone
would offer a sound the computer cannot select (or hide one it can). So both
routes build from :func:`entries` and add only their own columns.

THE PHONE'S BYTES (:func:`phone_audio`). The phone plays a notification sound
from a file it holds (iOS ``Library/Sounds``, Android a MediaStore row), at its
OWN notification volume (design §3.6: the computer's volume does not travel).
So it is served the sound at unity gain, as 16-bit WAV — the format both
platforms accept for a notification sound:

* a CUSTOM sound is the stored file itself (``custom_sounds`` already stores a
  normalised 16-bit WAV), byte for byte;
* a BUILT-IN ships as FLAC (package data), which neither platform's
  notification API plays, so it is decoded to WAV here. The decode is
  deterministic (same FLAC, same miniaudio → same bytes), and it is memoised
  per process, keyed by the catalog's own sha, so a list request costs at most
  one decode per built-in per daemon lifetime.

``sha256`` is ALWAYS the hash of the bytes the phone would receive, never of a
source file it will not see: the phone caches by that sha and checks it, so a
hash of anything else would make every fetch look corrupt.

THE WIRE NAME (:func:`push_stem`). A custom sound travels in a push as
``data.sound_custom = "c_" + sha256[:16]`` (ADR 0006 amendment 2026-11-04):
content-addressed, so the cloud never sees the user's sound name (P2) and the
name changes whenever the bytes do. The underscore is load-bearing — Android
``res/raw`` names allow no hyphen and the phone's file stem IS the wire value,
so there is no second mapping to drift. A built-in's push name is its id with
``-`` → ``_`` (the bundled file name the ADR's ``sound`` row maps to).

STDLIB-ONLY AT IMPORT, like ``notifications.sound``: miniaudio and the settings
registry load inside the functions that need them.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import re
from typing import Any

from local_operator.notifications import custom_sounds
from local_operator.notifications import sound as _sound

#: ``c_<sha256[:16]>`` — the frozen wire form of ``data.sound_custom`` and the
#: phone's cache/channel key (``c_<sha16>.wav``, ``sound.c_<sha16>``).
PUSH_STEM_RE = re.compile(r"^c_[0-9a-f]{16}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclasses.dataclass(frozen=True)
class Entry:
    """One listed sound, in picker order. ``custom`` is set for a custom row only."""

    id: str
    label: str
    builtin: bool
    #: ``None`` for ``system``, which has no audio of its own.
    duration_s: float | None
    custom: custom_sounds.CustomSound | None = None


def _labels() -> dict[str, str]:
    # The registry's keyed choices, resolved in the active `language`: the one
    # place a sound's display name is localized, so both surfaces show the same.
    from local_operator.settings_io import BY_KEY

    return {
        str(choice.value): str(choice.description)
        for choice in BY_KEY["notifications.sound.id"].resolved_choices
    }


def entries() -> list[Entry]:
    """Built-ins, then ``system``, then custom sounds oldest first.

    The same order as ``sound.sound_ids()`` and the settings choices. A custom
    sound's label is the user's own name (it is not translatable copy).
    """
    labels = _labels()
    rows = [
        Entry(entry.id, labels.get(entry.id, entry.id), True, entry.duration_s)
        for entry in _sound.builtin_sounds()
    ]
    # `system` is a row too (the settings spec lists 11 built-in rows): it has
    # no audio, which `duration_s: None` says, and every audio route 404s it.
    rows.append(
        Entry(
            _sound.SYSTEM_SOUND_ID,
            labels.get(_sound.SYSTEM_SOUND_ID, _sound.SYSTEM_SOUND_ID),
            True,
            None,
        )
    )
    rows.extend(
        Entry(row.id, row.name, False, row.duration_s, row) for row in custom_sounds.list_sounds()
    )
    return rows


def push_stem(sha256: str) -> str:
    """``c_<sha256[:16]>`` for a custom sound's content hash."""
    return f"c_{sha256[:16]}"


def builtin_push_name(sound_id: str) -> str:
    """The bundled file a built-in maps to on the phone (ADR 0006's ``sound`` row)."""
    return f"{sound_id.replace('-', '_')}.wav"


@functools.lru_cache(maxsize=16)
def _builtin_wav(sound_id: str, catalog_sha: str) -> tuple[bytes, str] | None:
    """A built-in decoded to unity-gain 16-bit WAV, and that WAV's sha256.

    Keyed by the catalog sha as well as the id so a reinstalled package with a
    re-exported sound can never be answered from a stale decode.
    """
    import os
    import tempfile

    import miniaudio

    entry = _sound.builtin(sound_id)
    if entry is None or entry.sha256 != catalog_sha or not entry.path.is_file():
        return None
    info = miniaudio.get_file_info(str(entry.path))
    decoded = miniaudio.decode_file(
        str(entry.path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=info.nchannels,
        sample_rate=info.sample_rate,
    )
    # miniaudio writes WAV only to a path; a private temp file, read and
    # removed in this call, keeps the result in memory and nothing on disk.
    descriptor, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(descriptor)
    try:
        miniaudio.wav_write_file(tmp, decoded)
        with open(tmp, "rb") as handle:
            data = handle.read()
    finally:
        os.unlink(tmp)
    return data, hashlib.sha256(data).hexdigest()


def phone_audio(sound_id: str) -> tuple[bytes, str] | None:
    """``(wav_bytes, sha256_of_those_bytes)`` for a built-in or custom id, else ``None``.

    ``sound_id`` is looked up in the catalog or the custom index and never
    joined to a path (``custom_sounds.resolve_file`` checks the id regex and
    resolved containment). The custom sha is recomputed from the bytes rather
    than trusted from the index, so a file changed behind the index can never
    be served under a sha it does not have.
    """
    entry = _sound.builtin(sound_id)
    if entry is not None:
        return _builtin_wav(entry.id, entry.sha256)
    found = custom_sounds.resolve_file(sound_id)
    if found is None:
        return None
    _row, path = found
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def phone_listing() -> dict[str, Any]:
    """The phone's list: every sound it could fetch, plus the current selection.

    ``sha256``/``size_bytes`` describe the bytes :func:`phone_audio` serves;
    ``push_name`` is the file name a push would name for this sound. ``system``
    has no audio, so it carries neither a sha nor a push name. The selection is
    read the way ``resolve`` reads it (``configured``), so the phone sees what
    the next notification obeys; ``sound_custom`` is the stem a push carries
    right now (``None`` unless a playable custom sound is selected and sound is
    on) — the one value the phone must hold to play the computer's sound.
    """
    sounds: list[dict[str, Any]] = []
    for entry in entries():
        row: dict[str, Any] = {
            "id": entry.id,
            "label": entry.label,
            "builtin": entry.builtin,
            "duration_s": entry.duration_s,
            "sha256": None,
            "size_bytes": None,
            "push_name": None,
        }
        if entry.custom is not None:
            path = custom_sounds.sound_path(entry.id)
            try:
                size = path.stat().st_size if path is not None and not path.is_symlink() else None
            except OSError:
                size = None
            if size is None:
                # Indexed but the file is gone: not fetchable, so not listed as
                # if it were (the audio route would 404 it).
                continue
            row["sha256"] = entry.custom.sha256
            row["size_bytes"] = size
            row["push_name"] = f"{push_stem(entry.custom.sha256)}.wav"
        elif entry.duration_s is not None:
            audio = phone_audio(entry.id)
            if audio is not None:
                row["sha256"] = audio[1]
                row["size_bytes"] = len(audio[0])
                row["push_name"] = builtin_push_name(entry.id)
        sounds.append(row)
    stored = _sound.configured()
    return {
        "sounds": sounds,
        "selected": stored.sound_id,
        "enabled": stored.enabled,
        "sound_custom": selected_push_stem(),
    }


def selected_push_stem(values: Any = None) -> str | None:
    """``data.sound_custom`` for the current settings, or ``None``.

    Present only when sound is ENABLED and the selection is a custom sound whose
    file exists — the conditions under which the phone's custom file is the
    right thing to play. A dangling custom id (file removed) answers ``None``:
    the push then carries only ``data.sound``'s soft-pulse fallback, which is
    exactly what the computer itself would fall back to.
    """
    stored = _sound.configured(values)
    if not stored.enabled or not custom_sounds.is_custom_id(stored.sound_id):
        return None
    found = custom_sounds.resolve_file(stored.sound_id)
    if found is None:
        return None
    row, _path = found
    if not _SHA256_RE.fullmatch(row.sha256):
        return None
    return push_stem(row.sha256)

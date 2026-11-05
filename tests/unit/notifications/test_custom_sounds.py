"""Custom notification sounds: the store, the import pipeline, path safety, the CLI.

WHAT IS PINNED, and why (notification-sounds design §3, settings spec C10):

* the caps refuse with their own codes: over 10 MB ``too_large``, over 10 s
  ``too_long`` (REFUSED, never truncated), a 51st sound ``limit_reached``;
* the magic sniff: Opus and M4A are ``unsupported_format`` with
  ``params.format``; text renamed ``.wav`` is ``undecodable``;
* the pipeline: silence is trimmed, loudness lands on the built-ins' -20 LUFS
  with the peak at or under -1 dBFS, ``keep_loudness`` leaves it alone;
* path safety: ids are generated in the ``custom-`` shape, a hostile id never
  reaches the filesystem, a symlinked store or sound file is refused;
* the selection: a custom id is a valid ``notifications.sound.id`` choice and
  resolves to its file; removing the selected sound resets to the default.

Every test runs in an isolated ``LOCAL_OPERATOR_CONFIG_DIR`` and builds its WAVs
with the stdlib ``wave`` module, so nothing depends on a codec on the host.
"""

from __future__ import annotations

import io
import math
import struct
import tracemalloc
import wave
from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.notifications import custom_sounds, sound
from local_operator.notifications.custom_sounds import SoundRejected


def tone(
    seconds: float,
    *,
    sample_rate: int = 44100,
    channels: int = 1,
    amplitude: float = 0.25,
    lead_silence: float = 0.0,
) -> bytes:
    """A sine as 16-bit WAV bytes, optionally preceded by digital silence."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        frames = bytearray()
        for i in range(int(lead_silence * sample_rate)):
            frames += b"\0\0" * channels
        for i in range(int(seconds * sample_rate)):
            value = int(32767 * amplitude * math.sin(2 * math.pi * 660 * i / sample_rate))
            frames += struct.pack("<h", value) * channels
        out.writeframes(bytes(frames))
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _code(raw: bytes, label: str = "clip.wav") -> SoundRejected:
    with pytest.raises(SoundRejected) as caught:
        custom_sounds.add(raw, source_name=label)
    return caught.value


# -- caps and formats ------------------------------------------------------------


def test_a_wav_is_added_and_stored_as_16_bit_wav(store: Path) -> None:
    row = custom_sounds.add(tone(0.5), source_name="chime.wav", name="My chime")
    assert custom_sounds.ID_RE.fullmatch(row.id) and row.id.startswith("custom-my-chime-")
    path = store / "sounds" / "custom" / f"{row.id}.wav"
    with wave.open(str(path)) as stored:
        assert stored.getsampwidth() == 2
    assert [s.id for s in custom_sounds.list_sounds()] == [row.id]
    assert row.source_name == "chime.wav" and row.normalized


def test_over_ten_megabytes_is_too_large() -> None:
    refusal = _code(b"RIFF" + b"\0" * (custom_sounds.MAX_INPUT_BYTES + 1))
    assert (refusal.code, refusal.status) == ("too_large", 413)


def test_over_ten_seconds_is_refused_not_truncated() -> None:
    refusal = _code(tone(10.5))
    assert (refusal.code, refusal.status) == ("too_long", 422)
    assert refusal.envelope()["params"]["seconds"] == "10.5"
    assert custom_sounds.list_sounds() == ()


def test_leading_silence_does_not_count_against_the_cap() -> None:
    row = custom_sounds.add(tone(9.0, lead_silence=3.0), source_name="padded.wav")
    assert row.duration_s == pytest.approx(9.0, abs=0.02)


@pytest.mark.parametrize(
    ("raw", "fmt"),
    [
        (b"OggS" + b"\0" * 24 + b"OpusHead" + b"\0" * 64, "opus"),
        (b"\0\0\0\x20ftypM4A " + b"\0" * 64, "m4a"),
    ],
)
def test_opus_and_m4a_are_unsupported_with_their_format(raw: bytes, fmt: str) -> None:
    refusal = _code(raw)
    assert (refusal.code, refusal.status) == ("unsupported_format", 415)
    assert refusal.envelope()["params"]["format"] == fmt


def test_text_renamed_wav_is_undecodable() -> None:
    refusal = _code(b"this is not audio\n" * 50)
    assert (refusal.code, refusal.status) == ("undecodable", 422)


def test_a_riff_header_with_garbage_is_undecodable() -> None:
    refusal = _code(b"RIFF\x24\0\0\0WAVEfmt " + b"\xff" * 200)
    assert refusal.code == "undecodable"


def test_the_fifty_first_sound_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(custom_sounds, "MAX_SOUNDS", 2)
    custom_sounds.add(tone(0.2), source_name="a.wav")
    custom_sounds.add(tone(0.2), source_name="b.wav")
    refusal = _code(tone(0.2))
    assert (refusal.code, refusal.status) == ("limit_reached", 422)


# -- loudness ----------------------------------------------------------------------


def _stored_channels(sound_id: str) -> tuple[list[list[float]], int]:
    path = custom_sounds.sound_path(sound_id)
    assert path is not None
    with wave.open(str(path)) as stored:
        rate, width = stored.getframerate(), stored.getnchannels()
        data = stored.readframes(stored.getnframes())
    samples = struct.unpack(f"<{len(data) // 2}h", data)
    return [[v / 32768 for v in samples[c::width]] for c in range(width)], rate


def test_import_matches_the_built_in_loudness_with_headroom() -> None:
    quiet = custom_sounds.add(tone(1.0, amplitude=0.02), source_name="quiet.wav")
    channels, rate = _stored_channels(quiet.id)
    assert custom_sounds.max_momentary_lufs(channels, rate) == pytest.approx(-20.0, abs=0.5)
    assert 20 * math.log10(max(abs(v) for v in channels[0])) <= -1.0 + 1e-3


def test_keep_loudness_leaves_the_level_alone() -> None:
    kept = custom_sounds.add(tone(1.0, amplitude=0.02), source_name="q.wav", keep_loudness=True)
    channels, _ = _stored_channels(kept.id)
    assert not kept.normalized
    assert max(abs(v) for v in channels[0]) == pytest.approx(0.02, abs=0.002)


def test_identical_stereo_folds_to_mono() -> None:
    row = custom_sounds.add(tone(0.5, channels=2), source_name="dual.wav")
    assert row.channels == 1


# -- path safety -------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    ["../config", "../../etc/passwd", "/etc/passwd", "custom-../x", "custom-a/b", "soft-pulse"],
)
def test_a_hostile_id_never_reaches_the_filesystem(hostile: str) -> None:
    assert custom_sounds.sound_path(hostile) is None
    assert custom_sounds.render(hostile, 70) is None or hostile == "soft-pulse"
    with pytest.raises(SoundRejected) as caught:
        custom_sounds.remove(hostile)
    assert caught.value.code in ("not_found", "builtin_readonly")


def test_a_hostile_upload_name_is_only_text(store: Path) -> None:
    row = custom_sounds.add(tone(0.3), source_name="../../evil.wav")
    assert row.source_name == "evil.wav"
    assert not (store.parent / "evil.wav").exists()
    assert sorted(p.name for p in (store / "sounds" / "custom").iterdir()) == sorted(
        ["index.json", f"{row.id}.wav"]
    )


def test_a_symlinked_sound_file_is_refused(store: Path, tmp_path_factory) -> None:
    row = custom_sounds.add(tone(0.3), source_name="a.wav")
    outside = tmp_path_factory.mktemp("outside") / "secret.wav"
    outside.write_bytes(tone(0.3))
    path = store / "sounds" / "custom" / f"{row.id}.wav"
    path.unlink()
    path.symlink_to(outside)
    assert custom_sounds.sound_path(row.id) is None
    assert custom_sounds.render(row.id, 70) is None
    assert sound.playable(row.id) is None


def test_a_symlinked_store_is_refused(store: Path, tmp_path_factory) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (store / "sounds").mkdir()
    (store / "sounds" / "custom").symlink_to(elsewhere, target_is_directory=True)
    refusal = _code(tone(0.3))
    assert refusal.code == "store_failed"
    assert list(elsewhere.iterdir()) == []


# -- selection ---------------------------------------------------------------------


def test_a_custom_id_is_a_setting_choice_and_resolves_to_its_file(store: Path) -> None:
    row = custom_sounds.add(tone(0.3), source_name="mine.wav", name="Mine")
    setting = settings_io.BY_KEY["notifications.sound.id"]
    choice = {c.value: c for c in setting.resolved_choices}[row.id]
    assert choice.description == "Mine"
    settings_io.write_setting(ConfigManager(store), setting, row.id)
    plan = sound.resolve(env={})
    assert (plan.kind, plan.sound_id) == ("file", row.id)
    assert plan.path == custom_sounds.sound_path(row.id)


def test_an_unknown_custom_id_is_refused_by_the_registry(store: Path) -> None:
    setting = settings_io.BY_KEY["notifications.sound.id"]
    with pytest.raises(ValueError):
        settings_io.write_setting(ConfigManager(store), setting, "custom-nothing-000000")


def test_removing_the_selected_sound_resets_to_the_default(store: Path) -> None:
    row = custom_sounds.add(tone(0.3), source_name="mine.wav")
    setting = settings_io.BY_KEY["notifications.sound.id"]
    settings_io.write_setting(ConfigManager(store), setting, row.id)
    removal = custom_sounds.remove(row.id)
    assert removal.selection_reset
    assert "Soft pulse is selected again" in removal.reset_text()
    assert sound.resolve(env={}).sound_id == sound.DEFAULT_SOUND_ID
    assert (
        ConfigManager(store)
        .get_config_value("notifications", {})
        .get("sound", {})
        .get("id", sound.DEFAULT_SOUND_ID)
        == sound.DEFAULT_SOUND_ID
    )
    assert custom_sounds.sound_path(row.id) is not None  # the safe path, now absent
    assert not custom_sounds.sound_path(row.id).exists()  # type: ignore[union-attr]


def test_removing_another_sound_keeps_the_selection(store: Path) -> None:
    keep = custom_sounds.add(tone(0.3), source_name="keep.wav")
    drop = custom_sounds.add(tone(0.3), source_name="drop.wav")
    settings_io.write_setting(
        ConfigManager(store), settings_io.BY_KEY["notifications.sound.id"], keep.id
    )
    assert not custom_sounds.remove(drop.id).selection_reset
    assert sound.resolve(env={}).sound_id == keep.id


def test_rename_validates_and_keeps_the_id() -> None:
    row = custom_sounds.add(tone(0.3), source_name="a.wav")
    renamed = custom_sounds.rename(row.id, "  Brass   ping ")
    assert (renamed.id, renamed.name) == (row.id, "Brass ping")
    for bad, code in (("", "invalid_name"), ("x" * 41, "invalid_name")):
        with pytest.raises(SoundRejected) as caught:
            custom_sounds.rename(row.id, bad)
        assert caught.value.code == code
    with pytest.raises(SoundRejected) as caught:
        custom_sounds.rename("soft-pulse", "Mine now")
    assert (caught.value.code, caught.value.status) == ("builtin_readonly", 409)


def test_render_applies_the_volume_curve_and_caches_by_sha() -> None:
    row = custom_sounds.add(tone(0.5), source_name="a.wav")
    full = custom_sounds.render(row.id, 100)
    seventy = custom_sounds.render(row.id, 70)
    assert full is not None and seventy is not None
    peaks = []
    for rendered in (full, seventy):
        with wave.open(str(rendered.path)) as out:
            data = out.readframes(out.getnframes())
        peaks.append(max(abs(v) for v in struct.unpack(f"<{len(data) // 2}h", data)))
    assert 20 * math.log10(peaks[1] / peaks[0]) == pytest.approx(20 * math.log10(0.49), abs=0.05)
    assert seventy.path.name.startswith(f"{row.id}-{row.sha256[:8]}-70")


# -- the CLI -------------------------------------------------------------------------


def test_cli_add_from_an_attachment_digest_selects_it(store: Path, capsys) -> None:
    from local_operator import sound_cli
    from local_operator.session.attachments import AttachmentStore

    ref = AttachmentStore().put_bytes(tone(0.4), "audio/wav")
    assert ref is not None
    assert (
        sound_cli.add_command(ref.digest, name="Brass ping", select=True, keep_loudness=False) == 0
    )
    out = capsys.readouterr().out
    row = custom_sounds.list_sounds()[0]
    assert row.source_name == ref.digest and row.name == "Brass ping"
    assert f"selected: notifications.sound.id = {row.id}" in out
    assert sound.resolve(env={}).sound_id == row.id


def test_cli_remove_of_the_selected_sound_says_so(store: Path, capsys) -> None:
    from local_operator import sound_cli

    clip = store / "c.wav"
    clip.write_bytes(tone(0.3))
    assert sound_cli.add_command(str(clip), name=None, select=True, keep_loudness=False) == 0
    row = custom_sounds.list_sounds()[0]
    capsys.readouterr()
    assert sound_cli.remove_command(row.id) == 0
    assert "Soft pulse is selected again" in capsys.readouterr().out


def test_cli_refusal_prints_the_code(store: Path, capsys) -> None:
    from local_operator import sound_cli

    clip = store / "notes.wav"
    clip.write_text("not audio\n" * 40)
    assert sound_cli.add_command(str(clip), name=None, select=False, keep_loudness=False) == 1
    assert "(undecodable)" in capsys.readouterr().err


# -- R1: a container that lies about its length -------------------------------


def _crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def _crc16(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x8005) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def constant_flac(seconds: float, *, declared_samples: int = 0, sample_rate: int = 48000) -> bytes:
    """A valid stereo FLAC of ``seconds`` of a quiet constant, with a CHOSEN header length.

    Every frame is two CONSTANT subframes (a few bytes each), so 20 minutes
    is ~220 KB, far under the 10 MB cap: the agent review's decode bomb, built
    here without an encoder. ``declared_samples=0`` is STREAMINFO's "length
    unknown", which ``flac_get_info`` reports as a duration of 0.0.
    """
    block = 4096
    info = struct.pack(">HH", block, block) + b"\0" * 6
    packed = (sample_rate << 44) | (1 << 41) | (15 << 36) | (declared_samples & ((1 << 36) - 1))
    info += packed.to_bytes(8, "big") + b"\0" * 16
    out = bytearray(b"fLaC" + bytes([0x80]) + len(info).to_bytes(3, "big") + info)
    subframe = b"\x00" + struct.pack(">h", 1200)
    for n in range(int(seconds * sample_rate) // block):
        # Fixed block size 4096 (0xC), 48 kHz (0xA), left/right (1), 16-bit.
        head = bytes([0xFF, 0xF8, 0xCA, 0x18]) + chr(n).encode("utf-8")
        head += bytes([_crc8(head)])
        body = head + subframe + subframe
        out += body + struct.pack(">H", _crc16(body))
    return bytes(out)


@pytest.mark.parametrize("declared_samples", [0, 48000], ids=["length-unknown", "claims-1s"])
def test_a_lying_header_cannot_make_the_decode_allocate(declared_samples: int) -> None:
    import miniaudio

    bomb = constant_flac(1200, declared_samples=declared_samples)
    assert len(bomb) < 300_000
    assert miniaudio.flac_get_info(bomb).duration <= 1.0  # the header lies
    tracemalloc.start()
    try:
        with pytest.raises(SoundRejected) as caught:
            custom_sounds.prepare(bomb, file_label="bomb.flac")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert caught.value.code == "too_long"
    # The whole 20 minutes would be ~460 MB of float32 alone; the bounded stream
    # holds at most PRECHECK_DURATION_S of it (~11.5 MB) plus one chunk.
    assert peak < 40 * 1024 * 1024, f"peak {peak / 1e6:.1f} MB"


def test_the_flac_fixture_decodes_when_it_is_short() -> None:
    """Control: the same generator at 2 s is a real, accepted sound."""
    row = custom_sounds.add(constant_flac(2.0), source_name="short.flac")
    assert row.duration_s > 1.5


def test_a_planted_symlink_in_a_sound_slot_is_unlinked_on_delete(store: Path) -> None:
    row = custom_sounds.add(tone(0.3), source_name="a.wav")
    slot = store / "sounds" / "custom" / f"{row.id}.wav"
    outside = store / "outside.wav"
    outside.write_bytes(tone(0.2))
    slot.unlink()
    slot.symlink_to(outside)
    custom_sounds.remove(row.id)
    assert not slot.is_symlink() and not slot.exists()
    assert outside.read_bytes() == tone(0.2)  # the target is untouched


def test_removing_the_selected_sound_logs_no_stray_warning(store: Path, caplog) -> None:
    row = custom_sounds.add(tone(0.3), source_name="mine.wav")
    settings_io.write_setting(
        ConfigManager(store), settings_io.BY_KEY["notifications.sound.id"], row.id
    )
    caplog.set_level("WARNING")
    assert custom_sounds.remove(row.id).selection_reset
    assert not [r for r in caplog.records if "does not carry" in r.getMessage()]


# -- QA round 2: the CLI's view of a custom selection, and its locale -----------


@pytest.mark.parametrize("verb", ["list", "doctor"])
def test_a_selected_custom_sound_is_not_reported_as_unknown(
    store: Path, capsys, monkeypatch, verb: str
) -> None:
    """Q5: `_problems` used `builtin()`, so every custom selection read as a typo."""
    from local_operator import sound_cli

    monkeypatch.delenv("LOCAL_OPERATOR_NO_SOUND", raising=False)
    # `doctor` returns early with banners off or over SSH, before the notes it
    # is meant to print; both are armed by the suite or may leak from a shell.
    # Only the env checks are lifted: no banner is raised by either verb.
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    for name in ("SSH_CONNECTION", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    row = custom_sounds.add(tone(0.3), source_name="mine.wav")
    settings_io.write_setting(
        ConfigManager(store), settings_io.BY_KEY["notifications.sound.id"], row.id
    )
    capsys.readouterr()
    assert (sound_cli.list_command() if verb == "list" else sound_cli.doctor_command()) == 0
    out = capsys.readouterr().out
    assert f"plays {row.id}" in out
    assert "is not a known sound" not in out


def test_an_unknown_custom_id_is_still_reported(store: Path, capsys, monkeypatch) -> None:
    """Control for Q5: a hand-edited id naming no stored sound still gets the note."""
    from local_operator import sound_cli
    from local_operator.notifications import sound as _sound

    monkeypatch.delenv("LOCAL_OPERATOR_NO_SOUND", raising=False)
    stored = _sound.configured({"notifications": {"sound": {"id": "custom-gone-000000"}}})
    plan = _sound.resolve({"notifications": {"sound": {"id": "custom-gone-000000"}}}, env={})
    assert any("is not a known sound" in p for p in sound_cli._problems(stored, plan))


def test_cli_refusals_render_in_the_active_locale(store: Path, capsys) -> None:
    """Q6: `Msg.envelope()` renders en by default; the CLI must pass the locale."""
    from local_operator import sound_cli
    from local_operator.i18n import text

    previous = text.active_locale()
    text.set_active_locale("fr")
    try:
        assert sound_cli.add_command("f" * 32, name=None, select=False, keep_loudness=False) == 1
        err = capsys.readouterr().err
        with pytest.raises(SoundRejected) as caught:
            custom_sounds.remove("soft-pulse")
        refusal = caught.value.text
    finally:
        text.set_active_locale(previous)
    assert "Aucune pièce jointe" in err, err
    assert "son intégré" in refusal, refusal
    # The route envelope stays en (the desktop keys on `code`).
    assert "built-in sound" in caught.value.envelope()["text"]


def test_pure_silence_is_its_own_code_not_undecodable(store: Path, capsys) -> None:
    """UI D2 QA r2 Q5: a silent clip decodes fine; it must not read as damaged."""
    from local_operator import sound_cli

    refusal = _code(tone(1.0, amplitude=0.0), "quiet.wav")
    assert (refusal.code, refusal.status) == ("silent", 422)
    assert refusal.envelope()["code"] == "silent"
    assert "is silent" in refusal.text
    clip = store / "quiet.wav"
    clip.write_bytes(tone(1.0, amplitude=0.0))
    assert sound_cli.add_command(str(clip), name=None, select=False, keep_loudness=False) == 1
    err = capsys.readouterr().err
    assert "quiet.wav is silent" in err and "(silent)" in err

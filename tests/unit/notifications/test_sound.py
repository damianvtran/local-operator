"""``notifications.sound`` and ``notifications.play``: plan, gain, throttle, receipt.

The player tests run the REAL decode-and-play path on miniaudio's NULL device
(``LOCAL_OPERATOR_SOUND_BACKEND=null``), which needs no speakers, and read the
receipt it writes after playback. The suite-wide ``LOCAL_OPERATOR_NO_SOUND``
arm is cleared per test only where the test is about playing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.notifications import play, sound


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(directory))
    return directory


def _values(**sound_values: object) -> dict[str, object]:
    return {"notifications": {"sound": sound_values}}


def test_the_default_plan_is_soft_pulse_at_seventy_percent() -> None:
    plan = sound.resolve({}, env={})
    assert (plan.kind, plan.sound_id, plan.gain) == ("file", "soft-pulse", 0.49)
    assert plan.path is not None and plan.path.name == "soft-pulse.flac"
    assert plan.mutes_host_sound


@pytest.mark.parametrize(
    ("volume", "expected"), [(0, 0.0), (50, 0.25), (70, 0.49), (100, 1.0), (150, 1.0), (-5, 0.0)]
)
def test_the_volume_curve_is_squared_and_clamped(volume: int, expected: float) -> None:
    assert sound.gain(volume) == expected


def test_a_non_number_volume_reads_as_the_default() -> None:
    assert sound.gain("loud") == sound.gain(True) == 0.49


def test_off_is_silent_and_still_mutes_the_host() -> None:
    plan = sound.resolve(_values(enabled=False), env={})
    assert plan.kind == "silent" and plan.mutes_host_sound


def test_system_plays_nothing_and_leaves_the_host_sound_alone() -> None:
    plan = sound.resolve(_values(id="system"), env={})
    assert plan.kind == "system" and not plan.mutes_host_sound


def test_volume_zero_is_silent() -> None:
    assert sound.resolve(_values(volume=0), env={}).kind == "silent"


def test_an_unknown_hand_edited_id_falls_back_to_the_default_sound() -> None:
    plan = sound.resolve(_values(id="nope"), env={})
    assert (plan.kind, plan.sound_id) == ("file", "soft-pulse")


def test_the_mute_switch_wins() -> None:
    plan = sound.resolve({}, env={sound.ENV_NO_SOUND: "1"})
    assert plan.kind == "silent"


def test_resolve_reads_the_config_file_fresh(config_dir: Path) -> None:
    config_dir.mkdir(parents=True)
    path = config_dir / "config.yml"
    path.write_text("values:\n  notifications:\n    sound:\n      id: hum-swell\n")
    assert sound.resolve(env={}).sound_id == "hum-swell"
    path.write_text(
        "values:\n  notifications:\n    sound:\n      id: bubble-pulse\n      volume: 50\n"
    )
    plan = sound.resolve(env={})
    assert (plan.sound_id, plan.gain) == ("bubble-pulse", 0.25)


def test_the_throttle_allows_one_sound_per_window(config_dir: Path) -> None:
    assert sound.claim_throttle(now=1000.0) is True
    assert sound.claim_throttle(now=1000.2) is False
    assert sound.claim_throttle(now=1001.49) is False
    assert sound.claim_throttle(now=1001.6) is True
    assert (config_dir / "run" / sound.THROTTLE_FILE).exists()


def test_the_registry_refuses_an_unknown_sound_id() -> None:
    from local_operator.settings_io import BY_KEY, validate

    setting = BY_KEY["notifications.sound.id"]
    assert validate(setting, "soft-pulse") is None
    assert validate(setting, "system") is None
    assert "expected one of" in (validate(setting, "bogus") or "")
    volume = BY_KEY["notifications.sound.volume"]
    assert validate(volume, 101) and validate(volume, -1) and validate(volume, 100) is None


def _receipts(config_dir: Path) -> list[dict[str, object]]:
    path = config_dir / "run" / sound.RECEIPT_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_null_backend_plays_the_file_and_writes_a_receipt(config_dir: Path) -> None:
    entry = sound.builtin("bubble-pulse")
    assert entry is not None
    env = {"LOCAL_OPERATOR_SOUND_BACKEND": "null"}
    code = play.play(entry.path, 0.49, sound_id=entry.id, sha256=entry.sha256, leg="t", env=env)
    assert code == 0
    [receipt] = _receipts(config_dir)
    assert receipt["sound_id"] == "bubble-pulse" and receipt["gain"] == 0.49
    assert receipt["backend"] == "null" and receipt["sha256"] == entry.sha256
    # 0.295 s at 44.1 kHz: proof the whole file went through the device.
    assert receipt["frames"] == 13009


def test_the_mute_switch_stops_the_player_before_any_device(config_dir: Path) -> None:
    entry = sound.builtin("bubble-pulse")
    assert entry is not None
    env = {"LOCAL_OPERATOR_NO_SOUND": "1", "LOCAL_OPERATOR_SOUND_BACKEND": "null"}
    assert play.play(entry.path, 0.49, env=env) == 0
    assert _receipts(config_dir) == []


def test_a_null_backend_failure_never_falls_back_to_afplay(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("no device")

    monkeypatch.setattr(play, "_play_miniaudio", broken)
    monkeypatch.setattr(play, "_play_afplay", lambda *_a: pytest.fail("reached speakers"))
    assert play.play(Path("x.flac"), 0.5, env={"LOCAL_OPERATOR_SOUND_BACKEND": "null"}) == 1


def test_the_gain_scales_the_samples() -> None:
    import array

    scaled = play._scaled(array.array("h", [10000, -10000, 32767]), 0.49)
    assert list(scaled) == [4900, -4900, 16056]


def test_the_player_child_runs_isolated_end_to_end(config_dir: Path) -> None:
    """The exact argv a banner spawns, run to completion under ``-I``."""
    plan = sound.resolve(_values(id="bubble-pulse"), env={})
    argv = sound.player_argv(plan, leg="detached")
    assert argv[:4] == [sys.executable, "-I", "-m", "local_operator.notifications.play"]
    env = {
        "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
        "LOCAL_OPERATOR_SOUND_BACKEND": "null",
        "HOME": str(config_dir.parent),
    }
    subprocess.run(argv, env=env, check=True, timeout=30)
    [receipt] = _receipts(config_dir)
    assert (receipt["sound_id"], receipt["leg"]) == ("bubble-pulse", "detached")


def test_play_plan_is_throttled(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[list[str]] = []
    monkeypatch.setattr(
        "local_operator.proc.spawn_detached", lambda argv, **_k: started.append(argv) or True
    )
    plan = sound.resolve({}, env={})
    assert sound.play_plan(plan, leg="osc") is True
    assert sound.play_plan(plan, leg="osc") is False
    assert len(started) == 1
    assert sound.play_plan(sound.resolve(_values(id="system"), env={}), leg="osc") is False

"""Legacy waveform repair: real MP3 decoding and fail-closed resource limits."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from local_operator.audiogen import recovery
from local_operator.session.attachments import AttachmentStore

FIXTURE = Path(__file__).parents[2] / "fixtures" / "audio" / "tone.mp3"


@pytest.fixture(autouse=True)
def clear_attempts():
    recovery._recover.cache_clear()
    yield
    recovery._recover.cache_clear()


def test_decoder_is_a_base_requirement_not_only_an_extra():
    """uv generation installs request the bare distribution, never [all]."""
    import tomllib

    from packaging.requirements import Requirement

    manifest = Path(__file__).parents[3] / "pyproject.toml"
    dependencies = tomllib.loads(manifest.read_text())["project"]["dependencies"]
    assert any(
        (requirement := Requirement(value)).name == "miniaudio" and requirement.marker is None
        for value in dependencies
    )


def test_real_mp3_recovered_once_and_persisted(tmp_path, monkeypatch):
    store = AttachmentStore(tmp_path)
    ref = store.put_bytes(FIXTURE.read_bytes(), "audio/mpeg")
    assert ref is not None
    run = Mock(wraps=recovery.subprocess.run)
    monkeypatch.setattr(recovery.subprocess, "run", run)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda _: recovery.waveform_payload(store, ref.digest), range(4))
        )
    assert all(payload == results[0] for payload in results)
    assert results[0] is not None and len(results[0]["peaks"]) == 240
    assert max(results[0]["peaks"]) > 0
    assert store.get_peaks(ref.digest) == results[0]
    assert run.call_count == 1


@pytest.mark.parametrize("failure", ["invalid", "timeout", "decoder_missing"])
def test_failed_recovery_is_cached_and_never_fakes_silence(tmp_path, monkeypatch, failure):
    store = AttachmentStore(tmp_path)
    ref = store.put_bytes(b"not audio", "audio/mpeg")
    assert ref is not None
    if failure == "timeout":
        run = Mock(side_effect=recovery.subprocess.TimeoutExpired("decode", 10))
    elif failure == "decoder_missing":
        run = Mock(return_value=recovery.subprocess.CompletedProcess([], 0, b"null"))
    else:
        run = Mock(wraps=recovery.subprocess.run)
    monkeypatch.setattr(recovery.subprocess, "run", run)
    assert recovery.waveform_payload(store, ref.digest) is None
    assert recovery.waveform_payload(store, ref.digest) is None
    assert store.get_peaks(ref.digest) is None
    assert run.call_count == 1
    assert run.call_args.kwargs["timeout"] == recovery.RECOVERY_TIMEOUT_SECONDS


def test_input_cap_and_non_audio_do_not_launch_codec(tmp_path, monkeypatch):
    store = AttachmentStore(tmp_path)
    run = Mock(side_effect=AssertionError("codec must not launch"))
    monkeypatch.setattr(recovery.subprocess, "run", run)
    monkeypatch.setattr(recovery, "MAX_RECOVERY_BYTES", 32)
    for raw, mime in [(b"x" * 33, "audio/mpeg"), (b"picture", "image/png")]:
        ref = store.put_bytes(raw, mime)
        assert ref is not None
        assert recovery.waveform_payload(store, ref.digest) is None
    run.assert_not_called()


def test_busy_decoder_has_a_bounded_lock_wait(tmp_path, monkeypatch):
    store = AttachmentStore(tmp_path)
    ref = store.put_bytes(FIXTURE.read_bytes(), "audio/mpeg")
    assert ref is not None
    run = Mock(side_effect=AssertionError("busy decoder must not launch another child"))
    monkeypatch.setattr(recovery.subprocess, "run", run)
    monkeypatch.setattr(recovery, "RECOVERY_TIMEOUT_SECONDS", 0.001)
    with recovery._RECOVERY_LOCK:
        assert recovery.waveform_payload(store, ref.digest) is None
    run.assert_not_called()

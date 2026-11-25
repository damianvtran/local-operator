"""Failure-path checks for the installed-wheel team upgrade smoke."""

from __future__ import annotations

import importlib
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("smoke_team_session_upgrade")


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="POSIX packaging smoke")
def test_failed_send_reaps_unregistered_detached_runtime_after_ingress(
    smoke, monkeypatch, tmp_path
):
    fixture = smoke.Device(tmp_path / "fixture", Path(sys.executable))
    other = smoke.Device(tmp_path / "other", Path(sys.executable))
    detached = {}
    wait = ("-c", "import time; time.sleep(60)")
    ingress = None
    reap = smoke.reap_owned

    def after_ingress(owner):
        assert ingress is not None and ingress.poll() is not None
        reap(owner)

    monkeypatch.setattr(smoke, "reap_owned", after_ingress)
    try:
        primary = smoke.SmokeError("stalled first send")
        with pytest.raises(smoke.SmokeError, match="stalled first send") as caught:
            with fixture.cleanup(), fixture.process("ingress", *wait) as ingress:
                for device in (fixture, other):
                    # Detached from ingress and deliberately no registry row.
                    detached[device] = subprocess.Popen(
                        [sys.executable, *wait],
                        env=device.env,
                        cwd=device.root,
                        start_new_session=True,
                    )
                    assert detached[device].pid in smoke.owned_processes(device.owner)
                raise primary
        assert caught.value is primary
        # The old frontend-group-only cleanup leaves this detached PID alive.
        assert detached[fixture].wait(timeout=1) == -signal.SIGKILL
        assert detached[other].poll() is None
    finally:
        for proc in detached.values():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


def test_cleanup_does_not_mask_the_original_failure(smoke):
    def failed_cleanup():
        raise RuntimeError("process still alive")

    original = ValueError("send failed")
    with pytest.raises(ValueError, match="send failed") as caught:
        with smoke.cleanup_after(failed_cleanup):
            raise original
    assert caught.value is original
    assert caught.value.__notes__ == ["Smoke teardown also failed: process still alive"]


def test_persisted_reply_requires_assistant_text_after_the_matching_user_turn(smoke, tmp_path):
    device = smoke.Device(tmp_path / "fixture", Path(sys.executable))
    directory = device.session_dir("smoke-session")
    directory.mkdir(parents=True)
    journal = directory / "transcript.jsonl"
    prompt = "first upgraded send"

    def row(role, text, *, kind="message", compact=False):
        return (
            json.dumps(
                {
                    "type": "message",
                    "payload": {
                        "kind": kind,
                        "role": role,
                        "content": [{"text": text} if compact else {"type": "text", "text": text}],
                    },
                }
            )
            + "\n"
        )

    reply = row("assistant", smoke.MOCK_REPLY)
    journal.write_text(
        reply
        + row("user", prompt, compact=True)
        + row("tool", smoke.MOCK_REPLY)
        + row("assistant", smoke.MOCK_REPLY, kind="custom")
        + reply[:10],
        encoding="utf-8",
    )
    assert not smoke.persisted_reply(device, directory.name, prompt)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(reply[10:])
    assert smoke.persisted_reply(device, directory.name, prompt)

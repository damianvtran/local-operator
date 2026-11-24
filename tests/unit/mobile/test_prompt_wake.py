"""The phone's cold prompt wake answers before the boot, and delivers once.

Measured on the operator's machine (wake supervisor, same ``engage_runtime``
path, Nov 21-23 2026): cold boots run p50 10.1 s, p90 39.1 s, p99 106.8 s, and
34% exceeded the 15 s ack the ``/command`` route used to hold the request for —
so a third of phone wakes answered 504 "session did not answer" while the
runtime kept booting, and the session dropped out of the active section.

These tests stand the real ``/command`` route up (Starlette ``TestClient`` in
its context-manager form, so the event loop and the daemon-owned wake task
outlive one request) with ``continue_command`` replaced by a boot that is held
open until the test releases it — the seam that stands in for a slow cold boot
without spending 40 real seconds.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import (
    PROMPT_WAKE_DEADLINE_S,
    MobileDaemon,
    build_app,
)

SESSION = "cold-wake"
COMMAND_ID = "12345678-1234-4678-9234-567812345678"
PASSWORD = "pw123"


def _durable_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    directory = cfg / "sessions" / SESSION
    directory.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    asyncio.run(Transcript(directory).append_message(Message.user("existing", id="existing")))
    return cfg


class _SlowBoot:
    """A ``continue_command`` whose runtime is "constructing" until released.

    ``release_after_s`` is a BACKSTOP, never the assertion: a build WITHOUT the
    fix awaits the whole boot inside the request, so it must fail on the
    structural check ("the answer arrived while the boot was still held")
    instead of hanging the suite.
    """

    def __init__(self, *, fail: bool = False, release_after_s: float = 10.0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.release = threading.Event()
        self.fail = fail
        self.release_after_s = release_after_s
        self.finished = threading.Event()

    async def __call__(self, config_root, command, *, origin="", **engage):  # noqa: ANN001, ANN003, ANN204
        self.calls.append({"command_id": command.command_id, "origin": origin, **engage})
        on_constructing = engage.get("on_constructing")
        if on_constructing is not None:
            on_constructing()
        deadline = time.monotonic() + self.release_after_s
        try:
            while not self.release.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            if self.fail:
                raise RuntimeError("could not start a runtime for session cold-wake: boom")
            return SimpleNamespace(close=lambda: None), "prompt admitted"
        finally:
            self.finished.set()


def _post(client: TestClient) -> Any:
    return client.post(
        f"/api/sessions/{SESSION}/command",
        json={"op": "prompt", "command_id": COMMAND_ID, "text": "wake up"},
    )


def _wait_until(predicate, timeout_s: float = 30.0) -> None:  # noqa: ANN001
    """A hang backstop, not an assertion: returns the moment ``predicate`` holds."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


def test_cold_prompt_answers_waking_before_the_boot_and_delivers_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance (fix 1): 2xx in < 1 s, admitted once, session stays active."""
    from local_operator.mobile import attach_client

    _durable_session(tmp_path, monkeypatch)
    boot = _SlowBoot()
    monkeypatch.setattr(attach_client, "continue_command", boot)
    daemon = MobileDaemon(port=0, password=PASSWORD)
    with TestClient(build_app(daemon)) as client:
        client.post("/login", data={"password": PASSWORD})

        response = _post(client)
        assert response.status_code == 200, response.text
        # Structural, not a clock (AGENTS.md "Wait on the event"): the answer
        # arrived while the boot was still held open.
        assert not boot.finished.is_set(), "the wake held the request for the whole boot"
        assert response.json() == {"ok": True, "detail": "session is waking", "waking": True}
        assert SESSION in daemon.table.provisional_active

        # The phone's reconnect retry carries the same command id: it joins the
        # in-flight wake rather than engaging (and delivering) a second time.
        retry = _post(client)
        assert retry.status_code == 200, retry.text
        assert retry.json()["waking"] is True
        assert not boot.finished.is_set()
        assert len(boot.calls) == 1

        boot.release.set()
        _wait_until(lambda: not daemon._prompt_wakes)
        assert len(boot.calls) == 1
        # The boot gets the generous wake deadline, not the 15 s ack.
        assert boot.calls[0]["deadline_s"] == PROMPT_WAKE_DEADLINE_S
        assert boot.calls[0]["origin"] == "mobile"
        # Success retains the claim for discovery to settle, as before.
        assert SESSION in daemon.table.provisional_active


def test_a_boot_that_fails_after_waking_still_reports_the_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed boot retracts the active claim and ends the session's stream."""
    from local_operator.mobile import attach_client

    _durable_session(tmp_path, monkeypatch)
    boot = _SlowBoot(fail=True)
    monkeypatch.setattr(attach_client, "continue_command", boot)
    daemon = MobileDaemon(port=0, password=PASSWORD)
    with TestClient(build_app(daemon)) as client:
        client.post("/login", data={"password": PASSWORD})
        frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        daemon.table.session_subscribers.setdefault(SESSION, set()).add(frames)

        response = _post(client)
        assert response.status_code == 200, response.text
        assert response.json()["waking"] is True

        boot.release.set()
        _wait_until(lambda: not daemon._prompt_wakes)
        assert SESSION not in daemon.table.provisional_active
        published: list[dict[str, Any]] = []
        while not frames.empty():
            published.append(frames.get_nowait())
        # The last frame the stream saw is the terminal ``ended`` receipt.
        assert published, "the stream was never told the wake failed"
        assert published[-1].get("ended") is True


def test_a_boot_that_fails_before_constructing_is_still_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No runtime ever constructing means the request itself carries the error."""
    from local_operator.mobile import attach_client

    _durable_session(tmp_path, monkeypatch)

    async def refuse(config_root, command, *, origin="", **engage):  # noqa: ANN001, ANN003, ANN202
        raise RuntimeError("not enough memory to start a session runtime")

    monkeypatch.setattr(attach_client, "continue_command", refuse)
    daemon = MobileDaemon(port=0, password=PASSWORD)
    with TestClient(build_app(daemon)) as client:
        client.post("/login", data={"password": PASSWORD})
        response = _post(client)
        assert response.status_code >= 400
        assert "waking" not in response.json()
        assert SESSION not in daemon.table.provisional_active
        assert not daemon._prompt_wakes


def test_capabilities_advertise_the_background_prompt_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clients learn the ``waking`` ack exists from the capabilities object."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    client = TestClient(build_app(MobileDaemon(port=0, password=PASSWORD)))
    client.post("/login", data={"password": PASSWORD})
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert response.json()["capabilities"]["prompt_wake"] == {
        "available": True,
        "background": True,
        "version": 1,
    }

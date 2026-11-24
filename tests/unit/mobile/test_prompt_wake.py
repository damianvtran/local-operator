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
import contextlib
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


# -- resume (fix 2) -----------------------------------------------------------


class _SlowStart:
    """A ``MobileDaemon._spawn_session`` whose child is spawned but still booting.

    The 10 s hold is a backstop for the same reason as ``_SlowBoot``'s: a
    build without the fix awaits the whole boot inside the request.
    """

    def __init__(self, *, fail: BaseException | None = None, spawn: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self.release = asyncio.Event()
        self.fail = fail
        self.spawn = spawn
        self.finished = False

    async def __call__(self, cwd, provider, model_id, session_id, effort=None, **start):  # noqa: ANN001, ANN003, ANN204
        self.calls.append({"session_id": session_id, **start})
        on_spawned = start.get("on_spawned")
        if self.spawn and on_spawned is not None:
            on_spawned(4242)
        try:
            if self.spawn:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.release.wait(), timeout=10.0)
            if self.fail is not None:
                raise self.fail
            return 4242
        finally:
            self.finished = True


async def _resume_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: _SlowStart):  # noqa: ANN202
    import httpx

    from local_operator.harness.types import Message
    from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
    from local_operator.session.transcript import Transcript

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = tmp_path / "sessions" / SESSION
    directory.mkdir(parents=True)
    await Transcript(directory).append_message(Message.user("existing", id="existing"))
    daemon = MobileDaemon(password="pw")
    monkeypatch.setattr(daemon, "_spawn_session", start)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app(daemon)),
        base_url="http://fixture",
        cookies={COOKIE_NAME: sign_cookie("pw")},
    )
    return daemon, client


async def _settle(daemon: MobileDaemon) -> None:
    for _ in range(300):
        if not daemon._session_starts and not daemon._early_resumes:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the start never settled")


@pytest.mark.asyncio
async def test_resume_answers_once_the_child_is_spawned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance (fix 2): a booting resume is a 2xx with the pid, not a 30 s 500."""
    start = _SlowStart()
    daemon, client = await _resume_rig(tmp_path, monkeypatch, start)
    async with client:
        first, second = await asyncio.gather(
            client.post("/api/sessions/resume", json={"session_id": SESSION}),
            client.post("/api/sessions/resume", json={"session_id": SESSION}),
        )
    assert first.status_code == 200, first.text
    # Both answers arrived while the boot was still held open.
    assert not start.finished, "the resume held the request for the whole boot"
    expected = {"ok": True, "pid": 4242, "session_id": SESSION, "starting": True}
    assert first.json() == expected
    assert second.json() == expected
    # Coalesced onto ONE start, bounded by the boot deadline, not the 30 s ack.
    assert len(start.calls) == 1
    assert start.calls[0]["ready_timeout_s"] == PROMPT_WAKE_DEADLINE_S
    start.release.set()
    await _settle(daemon)


@pytest.mark.asyncio
async def test_a_resume_that_fails_after_starting_ends_the_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = _SlowStart(fail=RuntimeError("session exited before becoming ready"))
    daemon, client = await _resume_rig(tmp_path, monkeypatch, start)
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    daemon.table.session_subscribers.setdefault(SESSION, set()).add(frames)
    async with client:
        response = await client.post("/api/sessions/resume", json={"session_id": SESSION})
    assert response.status_code == 200, response.text
    assert response.json()["starting"] is True
    start.release.set()
    await _settle(daemon)
    published = [frames.get_nowait() for _ in range(frames.qsize())]
    assert published, "the stream was never told the resume failed"
    assert published[-1].get("ended") is True


@pytest.mark.asyncio
async def test_a_resume_timeout_says_what_happened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``str(TimeoutError())`` is empty; the phone must get a real sentence."""
    start = _SlowStart(fail=TimeoutError(), spawn=False)
    daemon, client = await _resume_rig(tmp_path, monkeypatch, start)
    async with client:
        response = await client.post("/api/sessions/resume", json={"session_id": SESSION})
    assert response.status_code == 504, response.text
    assert response.json() == {"error": "The session took too long to start. Try again."}
    await _settle(daemon)

"""``/v1/desktop/operator/setup`` through the real bearer gate (design v2 §6, C3).

The ASGI app mounts the production router with NO dependency overrides, so the
bearer/origin admission is the real one. Only the OS prompt is stubbed (the person
answering it) plus the anchor-root seam, as in the module's own tests.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from local_operator.operator import elevate, handlers, trust
from local_operator.operator.elevate import ElevationResult
from local_operator.server.features import DESKTOP_ONLY_FEATURES, feature_flags
from local_operator.server.routes import desktop_operator_setup as routes

TOKEN = "fixture-desktop-bearer"
ORIGIN = "http://desktop.fixture.test"
PATH = "/v1/desktop/operator/setup"


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", ORIGIN)
    anchor_root = tmp_path / "anchors"
    anchor_root.mkdir()
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", anchor_root)
    monkeypatch.setattr(handlers, "_is_privileged", lambda: False)
    real_open = trust._open_no_follow

    def as_root(path: Path) -> Any:
        opened = real_open(path)
        if opened is None:
            return None
        descriptor, _ = opened
        return descriptor, SimpleNamespace(st_uid=0, st_mode=0o100644)

    monkeypatch.setattr(trust, "_open_no_follow", as_root)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(trust, "_on_windows", lambda: False)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *a, **k: (
            f"/usr/bin/{name}" if name in ("pkexec", "sudo") else real_which(name)
        ),
    )
    prompts: list[dict[str, Any]] = []

    def approve(command: list[list[str]], **kwargs: Any) -> ElevationResult:
        prompts.append(kwargs)
        trust.anchor_path().write_bytes(trust.staging_path(root).read_bytes())
        return ElevationResult(0)

    monkeypatch.setattr(elevate, "run_elevated", approve)
    app = FastAPI()
    app.include_router(routes.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN},
    ) as client:
        yield SimpleNamespace(client=client, app=app, root=root, prompts=prompts)


@pytest.mark.asyncio
async def test_off_then_on_with_one_gesture_prompt(desktop: Any) -> None:
    state = (await desktop.client.get(PATH)).json()["result"]
    assert state["state"] == "off" and state["can_prompt"] is True and state["blocked"] is None

    response = await desktop.client.post(PATH, json={})

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["outcome"] == "on" and result["actor"] == "desktop"
    assert result["message"]["code"] == "on"
    assert "Each new phone will still ask for your computer's password once." in (
        result["message"]["text"]
    )
    # The server never offers terminal sudo, and the OS prompt names this purpose.
    assert desktop.prompts == [{"purpose": "phone_approvals", "terminal_sudo": False}]
    # §6 item 5 / U18: a file-only key is reported, with the plain sentence.
    if result["file_only"]:
        assert result["file_only_notice"]["text"] == (
            "Approvals here won't ask for a fingerprint or password."
        )
    after = (await desktop.client.get(PATH)).json()["result"]
    assert after["state"] == "on" and after["last_attempt"]["outcome"] == "on"
    again = (await desktop.client.post(PATH, json={})).json()["result"]
    assert again["outcome"] == "already_on" and len(desktop.prompts) == 1


@pytest.mark.asyncio
async def test_unauthorized_and_closed_body(desktop: Any) -> None:
    anonymous = await desktop.client.post(PATH, json={}, headers={"Authorization": ""})
    assert anonymous.status_code in (401, 403)
    wrong = await desktop.client.post(PATH, json={}, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code in (401, 403)
    # §8 E2E negative control: no `sudo_secret` (or any other field) is accepted.
    secret = await desktop.client.post(PATH, json={"sudo_secret": "ADMIN_PASSWORD"})
    assert secret.status_code == 422
    assert desktop.prompts == [] and not trust.anchor_path().exists()


@pytest.mark.asyncio
async def test_privileged_server_is_refused(desktop: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handlers, "_is_privileged", lambda: True)

    state = (await desktop.client.get(PATH)).json()["result"]
    assert state["can_prompt"] is False and state["blocked"] == "privileged_process"
    response = await desktop.client.post(PATH, json={})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "privileged_process"
    assert desktop.prompts == [] and not trust.anchor_path().exists()


@pytest.mark.asyncio
async def test_linux_server_without_pkexec_is_refused(
    desktop: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: "/usr/bin/sudo" if name == "sudo" else None
    )

    state = (await desktop.client.get(PATH)).json()["result"]
    assert state["blocked"] == "no_password_prompt"
    assert state["message"]["text"] == (
        "This computer can't show a password prompt, so approvals can't be turned on from here."
    )
    response = await desktop.client.post(PATH, json={})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_password_prompt"
    assert desktop.prompts == []


@pytest.mark.asyncio
async def test_cancelled_prompt_reads_finish_setup(
    desktop: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        elevate, "run_elevated", lambda command, **k: ElevationResult(126, cancelled=True)
    )

    result = (await desktop.client.post(PATH, json={})).json()["result"]

    assert result["outcome"] == "cancelled" and result["ok"] is False
    assert (await desktop.client.get(PATH)).json()["result"]["state"] == "finish_setup"


@pytest.mark.asyncio
async def test_second_click_while_prompt_open_is_409(
    desktop: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    entered, release = threading.Event(), threading.Event()

    def slow(command: list[list[str]], **kwargs: Any) -> ElevationResult:
        entered.set()
        release.wait(5)
        return ElevationResult(126, cancelled=True)

    monkeypatch.setattr(elevate, "run_elevated", slow)
    first = asyncio.create_task(desktop.client.post(PATH, json={}))
    await asyncio.to_thread(entered.wait, 5)
    assert (await desktop.client.get(PATH)).json()["result"]["in_progress"] is True
    second = await desktop.client.post(PATH, json={})
    release.set()
    assert second.status_code == 409 and second.json()["detail"]["code"] == "in_progress"
    assert (await first).json()["result"]["outcome"] == "cancelled"


def test_capability_flag_is_published_desktop_only() -> None:
    assert feature_flags()["operator_setup"] == 1
    assert "operator_setup" in DESKTOP_ONLY_FEATURES

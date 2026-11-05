"""Real loopback HTTP -> AttachedSession -> authenticated runtime -> registry."""

import asyncio
import contextlib
import socket
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException

from tests.unit.session.test_account_profile_consent import stores as consent_stores
from tests.unit.session.test_profile_actions import identity, saved

stores = consent_stores


@pytest.mark.asyncio
async def test_real_desktop_profile_status_and_answer_without_conversational_gate(
    stores, tmp_path, monkeypatch
):
    from local_operator.harness.approval import mint_operator_cap, remember_operator_cap
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.server.routes import desktop_sessions as routes
    from local_operator.session.attached import AttachedSession
    from tests.unit.session.runtime import test_approval_authority_seam as seam

    session, action, _, saved_result = await saved(stores, receipt=True)
    monkeypatch.setattr(seam, "_AttachableSession", lambda: session)
    cap = mint_operator_cap()
    live = await seam._serve(tmp_path, operator_cap=cap)
    remember_operator_cap(live.record.pid, cap)
    wire = AttachClient(lambda _: None, lambda _: None, surface="desktop")
    await wire.connect(live.record, live.record.session_id)
    remote = AttachedSession(
        config_dir=tmp_path,
        session_id=live.record.session_id,
        takeover_factory=lambda: None,
        surface="desktop",
    )
    remote._client = wire
    epoch = session.frontend_state.epoch
    remote._frontend_store = SimpleNamespace(state=SimpleNamespace(epoch=epoch))
    sid = live.record.session_id

    @contextlib.asynccontextmanager
    async def bind(requested, **kwargs):
        if requested != sid:
            raise HTTPException(404, "unknown fixture session")
        yield SimpleNamespace(remote=remote)

    app = FastAPI()
    app.state.desktop_sessions = SimpleNamespace(session=bind)
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.include_router(routes.router)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "synthetic-profile-actions")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
            headers={"Authorization": "Bearer synthetic-profile-actions"},
        ) as http:
            url = f"/v1/desktop/sessions/{sid}"
            status = await http.post(url + "/profile-action-status", json=identity(action))
            assert status.status_code == 200, status.text
            assert status.json()["result"]["state"] == "available"
            denied = await http.post(
                url + "/profile-action-status",
                json=identity(action),
                headers={"Authorization": "Bearer bad"},
            )
            assert denied.status_code in (401, 403)
            origin = await http.post(
                url + "/profile-action-status",
                json=identity(action),
                headers={"Origin": "https://untrusted.example"},
            )
            assert origin.status_code == 403
            wrong_session = await http.post(
                "/v1/desktop/sessions/other/profile-action-status", json=identity(action)
            )
            assert wrong_session.status_code == 404
            unknown = await http.post(
                url + "/answers",
                json={"epoch": epoch, "request_id": "prv_" + "A" * 32, "approved": True},
            )
            assert unknown.status_code == 409
            bad = await http.post(
                url + "/profile-action-status", json={**identity(action), "profile_name": "other"}
            )
            assert bad.json()["result"]["state"] == "unavailable"
            body = {"epoch": epoch, "request_id": action["action_id"], "approved": True}
            wrong_epoch = await http.post(url + "/answers", json={**body, "epoch": "old"})
            assert wrong_epoch.status_code == 409
            remembered = await http.post(url + "/answers", json={**body, "remember": True})
            assert remembered.status_code in (409, 422)

            async def no_signer(*args, **kwargs):
                return None

            with monkeypatch.context() as no_authority:
                no_authority.setattr(wire, "_operator_cap", None)
                no_authority.setattr(wire, "_operator_signature", no_signer)
                drive = await http.post(url + "/answers", json=body)
                assert drive.status_code == 422, drive.text
            answered = await http.post(url + "/answers", json=body)
            assert answered.status_code == 200, answered.text
            assert answered.json()["result"]["detail"] == "succeeded"
            duplicate = await http.post(url + "/answers", json=body)
            assert duplicate.status_code == 200, duplicate.text
            final = await http.post(url + "/profile-action-status", json=identity(action))
            assert final.json()["result"]["state"] == "succeeded"
            # Preserve the actual loopback responses, never tokens/proofs.
            import json
            import os
            from pathlib import Path

            evidence = Path(os.environ.get("PROFILE_R3_EVIDENCE", tmp_path))
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / "desktop-action-http.json").write_text(
                json.dumps(
                    {
                        "saved_tool_result": saved_result.model_dump(mode="json"),
                        "source_session_id": sid,
                        "status": status.json(),
                        "answer": answered.json(),
                        "duplicate": duplicate.json(),
                        "final": final.json(),
                        "negative_statuses": {
                            "auth": denied.status_code,
                            "origin": origin.status_code,
                            "session": wrong_session.status_code,
                            "epoch": wrong_epoch.status_code,
                            "remember": remembered.status_code,
                            "unknown_action": unknown.status_code,
                            "no_operator_authority": drive.status_code,
                        },
                    },
                    indent=2,
                )
            )

            from tests.unit.session.test_account_profile_consent import (
                SEED_INSTRUCTIONS,
            )

            row = stores["account"].get_agent_by_name("reviewer")
            assert stores["account"].get_agent_system_prompt(row.id) == SEED_INSTRUCTIONS
    finally:
        server.should_exit = True
        await task
        sock.close()
        wire.close()
        await live.close(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("machine", [True, False])
async def test_agent_exec_input_is_not_a_human_prompt_but_later_ui_is(
    tmp_path, monkeypatch, machine
):
    from local_operator import exec_session, headless_print
    from local_operator.session.runtime.exec_control import start_exec_control
    from tests.unit.session.runtime.test_exec_control import FakeSession

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tools.builtin import _sessions_open_env

    marker = _sessions_open_env(None)["LOCAL_OPERATOR_AGENT_SHELL"] if machine else "0"
    monkeypatch.setenv("LOCAL_OPERATOR_AGENT_SHELL", marker)
    seen = []

    class InputSession(FakeSession):
        async def prompt(
            self, text, images=None, *, owner_typed=True, message_id=None, admitted=None, **kwargs
        ):
            seen.append((text, owner_typed))
            if admitted is not None and not admitted.done():
                admitted.set_result(None)

    session = InputSession()

    async def print_boundary(_session, prompts, **kwargs):
        assert await kwargs["prompt_handler"](prompts[0])
        await kwargs["before_dispose"]()
        return 0

    monkeypatch.setattr(headless_print, "run_print_mode", print_boundary)
    args = SimpleNamespace(yolo=False, json_mode=False, loop=None, loop_goal=None)
    assert await exec_session.run_session(session, "Update reviewer instructions.", args, None) == 0
    assert seen == [("Update reviewer instructions.", not machine)]
    # The exec-origin bit belongs to that queued input, not to the daemon.
    control = await start_exec_control(session, cwd=str(tmp_path))
    try:
        await control.handle.prompt("A later human message", owner_typed=True, wait_complete=True)
        assert seen[-1] == ("A later human message", True)
    finally:
        await control.aclose()

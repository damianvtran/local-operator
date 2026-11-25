"""Automatic auth-route selections through owner, transcript, and sign-in paths."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio

from local_operator.config import ConfigManager
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEvent,
)
from local_operator.model.configure import build_model_spec, configure_model
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import client_for_spec
from local_operator.providers.failover import stream_with_failover
from local_operator.session.model_selection import read_model_selection
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.session_factory import resolve_hosting_model_with_source
from tests.unit.providers.test_vertex_client import _sse_body


class _RouteRig:
    """Record real provider-client requests, replacing only the vendor transport."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = ConfigManager(root)
        self.store = AuthStore(root / "auth.db", config_dir=root)
        self.wire: list[httpx.Request] = []
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self._reply))

    def _reply(self, request: httpx.Request) -> httpx.Response:
        self.wire.append(request)
        body = (
            _sse_body()
            if "googleapis.com" in request.url.host
            else b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    def sign_in(self, provider: str) -> int:
        oauth = self.store.upsert_credential(
            provider,
            {
                "type": "oauth",
                "access": "synthetic-oauth",
                "project_id": "test-project",
                "location": "eu",
            },
        )
        self.store.upsert_credential(
            provider, {"type": "api_key", "key": "synthetic-key", "source": "login"}
        )
        return oauth.id

    def owner(self, model: ModelSpec, name: str, *, source: str = "config") -> Session:
        async def stream(
            request: ChatRequest, signal: AbortSignal | None
        ) -> AsyncIterator[StreamEvent]:
            async for event in stream_with_failover(
                request,
                self.store,
                None,
                lambda spec: client_for_spec(spec, http_client=self.http),
                session_id=name,
                signal=signal,
            ):
                yield event

        return Session(
            model=model,
            model_source=source,
            stream_fn=stream,
            tools=[],
            transcript=Transcript(self.root / "sessions" / name),
            system_blocks_provider=lambda: [],
            cwd=str(self.root),
        )

    async def assert_vertex_key_turn(self, owner: Session, expected: str) -> None:
        assert not self.wire
        await owner.prompt("after the Cloud route was blocked")
        assert len(self.wire) == 1
        wire = self.wire[0]
        assert wire.headers["x-goog-api-key"] == "synthetic-key"
        assert wire.url.path == f"/v1/publishers/google/models/{expected}:streamGenerateContent"


@pytest_asyncio.fixture
async def route_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_RouteRig]:
    from local_operator.paths import CONFIG_DIR_ENV

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    rig = _RouteRig(tmp_path)
    try:
        yield rig
    finally:
        await rig.http.aclose()
        rig.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["automatic", "same-id-pin", "legacy"])
async def test_auth_route_default_survives_real_journal_resume(
    route_rig: _RouteRig,
    selection: str,
) -> None:
    rig = route_rig
    cloud_id, key_id = "k3.1-pro", "kimi-k3.1-pro"
    oauth_id = rig.sign_in("kimi")
    first = rig.owner(configure_model("kimi", "", config_dir=rig.root).spec, "route-resume")
    directory = first.transcript.directory
    assert first.model_label == f"kimi/{cloud_id}"
    try:
        rig.store.block_credential(oauth_id, "kimi")
        await first.prompt("initial turn on the fallback credential")
        if selection == "same-id-pin":
            # Explicitly picking the previewed ID must persist even though
            # the selected label is unchanged.
            first.set_model(build_model_spec("kimi", cloud_id), explicit=True)
    finally:
        await first.dispose()
    if selection == "legacy":
        await Transcript(directory).append_custom(
            "selected_model", {"version": 2, "selector": f"kimi/{cloud_id}"}
        )
    saved = read_model_selection(directory)
    assert saved is not None and saved.selector == f"kimi/{cloud_id}"
    provider, model_id, source = resolve_hosting_model_with_source(
        None,
        argparse.Namespace(resume=directory.name),
        rig.config,
    )
    assert source == "resume"
    resumed = rig.owner(
        configure_model(provider, model_id, config_dir=rig.root).spec, directory.name, source=source
    )
    try:
        await resumed.prompt("first resumed turn")
        assert len(rig.wire) == 2
        assert [request.url.host for request in rig.wire] == ["api.moonshot.cn"] * 2
        assert [json.loads(request.content)["model"] for request in rig.wire] == [
            key_id,
            key_id if selection == "automatic" else cloud_id,
        ]
    finally:
        await resumed.dispose()


@pytest.mark.asyncio
async def test_tui_login_rehome_keeps_automatic_route_selection(route_rig: _RouteRig) -> None:
    from local_operator.providers.controller import ProviderController
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.conftest import await_session_boot

    rig = route_rig
    rig.config.update_config({"hosting": "radient", "model_name": "auto"})
    owner = rig.owner(build_model_spec("radient", "auto"), "route-tui")

    async def factory() -> Session:
        return owner

    app = OperatorApp(factory, provider_controller=ProviderController(rig.store, rig.root))
    try:
        async with app.run_test(size=(100, 30)):
            await await_session_boot(app)
            oauth_id = rig.sign_in("google-vertex")
            assert app._apply_login_defaults("google-vertex")
            assert not ConfigManager(rig.root).get_config_value("model_name")
            assert app._rehome_stranded_session("google-vertex") == (
                "Switched to google-vertex/gemini-4-neon — not signed in to radient."
            )
            assert owner.model_label == "google-vertex/gemini-4-neon"
            rig.store.block_credential(oauth_id, "google-vertex")
            await rig.assert_vertex_key_turn(owner, "gemini-2.5-flash")
    finally:
        await owner.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["desktop-auto", "desktop-pin", "attached-tui"])
async def test_owner_control_rehome_keeps_automatic_route_selection(
    route_rig: _RouteRig,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    from local_operator.server.utils.desktop_rehome import rehome_after_login
    from local_operator.server.utils.desktop_sessions import DesktopSessions
    from local_operator.session.attached import AttachedSession
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle

    rig = route_rig
    configured_id = "gemini-4-neon" if entry == "desktop-pin" else ""
    rig.config.update_config({"hosting": "google-vertex", "model_name": configured_id})
    owner = rig.owner(build_model_spec("radient", "auto"), "route-desktop")
    handle = ServingSessionHandle(
        owner,
        asyncio.get_running_loop(),
        cwd=str(rig.root),
        config_dir=rig.root,
        install_gates=False,
    )
    runtime = RuntimeServer(handle, kind="desktop")

    class BoundConnection:
        async def rehome_if_current(self, expected: str, provider: str, model_id: str) -> str:
            # Use the runtime's real validation and dispatch; no socket or discovery
            # process is needed for the pool's already-bound connection.
            return str(
                await runtime._dispatch(
                    "rehome_if_current",
                    {
                        "op": "rehome_if_current",
                        "expected": expected,
                        "provider": provider,
                        "model_id": model_id,
                    },
                )
            )

    pool = DesktopSessions(rig.root)
    cast(dict[str, Any], pool.bridges)["route-desktop"] = SimpleNamespace(
        session_id="route-desktop",
        remote_row=None,
        _owner_connection=BoundConnection,
        remote=SimpleNamespace(
            is_cold=False, canonical_current=True, frontend_state=owner.frontend_state
        ),
    )
    app = SimpleNamespace(state=SimpleNamespace(config_manager=rig.config, desktop_sessions=pool))
    try:
        oauth_id = rig.sign_in("google-vertex")
        if entry == "attached-tui":
            viewer = AttachedSession(
                config_dir=rig.root,
                session_id="route-desktop",
                takeover_factory=lambda: None,
            )
            monkeypatch.setattr(viewer, "_client", BoundConnection())
            await viewer.rehome_to_auth_default("radient/auto", "google-vertex")
            assert owner.model_label == "google-vertex/gemini-4-neon"
        else:
            assert await rehome_after_login(app, "google-vertex") == (
                [("radient/auto", "google-vertex/gemini-4-neon")],
                [],
            )
        rig.store.block_credential(oauth_id, "google-vertex")
        await rig.assert_vertex_key_turn(owner, configured_id or "gemini-2.5-flash")
    finally:
        await handle.dispose()

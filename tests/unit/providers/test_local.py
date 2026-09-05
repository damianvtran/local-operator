"""Local-provider contracts over an owned loopback server, never installed runtimes.

These tests prove the assembled HTTP/config/auth paths without downloading model
weights or touching another developer's Ollama/LM Studio listener.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from local_operator.config import ConfigManager
from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolResult,
)
from local_operator.model.configure import build_model_spec
from local_operator.model.discovery import available_models, cached_available_models
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.controller import ProviderController
from local_operator.providers.local import (
    LOCAL_PRESETS,
    model_overrides,
    normalize_base_url,
    resolve_base_url,
)
from local_operator.providers.local_discovery import discover_local
from local_operator.providers.oauth.callback_server import LoginCallbacks, LoginError
from local_operator.providers.registry import (
    get_provider_definition,
    list_login_providers,
)
from local_operator.settings_io import BY_KEY, validate, write_setting


@pytest.fixture
def local_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    state = {
        "status": 200,
        "models": [{"id": "model/with:tag", "max_model_len": 8192}],
        "requests": [],
        "native": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args):
            pass

        def do_GET(self):
            gate = state.get("gate")
            if gate is not None:
                gate.wait(timeout=10)
            state["requests"].append((self.path, self.headers.get("Authorization")))
            if self.path.endswith("/api/v1/models"):
                payload = state["native"]
                status = 200 if payload is not None else 404
            elif self.path.endswith("/v1/models"):
                payload = {"data": state["models"]}
                status = state["status"]
            else:
                payload, status = {}, 404
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            state["requests"].append((self.path, body))
            chunks = [
                {
                    "id": "local-turn",
                    "choices": [{"delta": {"reasoning_content": "Private protocol "}}],
                },
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "continuation", "content": "LOCAL_OK"},
                            "finish_reason": "stop",
                        }
                    ]
                },
                {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4}},
            ]
            wire = (
                "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire.encode())))
            self.end_headers()
            self.wfile.write(wire.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["endpoint"] = f"http://127.0.0.1:{server.server_port}/proxy/v1"
    state["manager"] = ConfigManager(tmp_path / "config")
    state["manager"].update_config({"hosting": ""})
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "value",
    [
        "ftp://host",
        "http://",
        "http://host:bad",
        "http://user:pass@host",
        "http://host?token=secret",
        "http://host/#secret",
        "http://host/v1/models",
        "http://host:0",
        "http://ho st",
    ],
)
def test_invalid_endpoint(value):
    with pytest.raises(ValueError):
        normalize_base_url(value)


def test_presets_and_settings_share_registry():
    providers = {p.id for p in list_login_providers()}
    for provider in LOCAL_PRESETS:
        assert provider in providers
        definition = get_provider_definition(provider)
        assert definition is not None and definition.local_setup
        assert f"providers.{provider}.base_url" in BY_KEY
        assert f"providers.{provider}.models" in BY_KEY
    assert normalize_base_url("https://host/proxy/v1/v1/") == "https://host/proxy/v1"


@pytest.mark.parametrize(
    "value",
    [
        "[]",
        '{"m":{"context_window":0}}',
        '{"m":{"supports_tools":"false"}}',
        '{"m":{"made_up":true}}',
    ],
)
def test_model_override_validation(value):
    with pytest.raises(ValueError):
        model_overrides(value)
    assert validate(BY_KEY["providers.lmstudio.models"], value)


def test_endpoint_cache_and_active_context_override(local_server):
    state = local_server
    manager = state["manager"]
    first = state["endpoint"]
    manager.update_config(
        {
            "providers": {
                "vllm": {
                    "base_url": first,
                    "models": '{"model/with:tag":{"context_window":32768,"supports_tools":false}}',
                }
            }
        }
    )
    rows, status = available_models("vllm", api_key=None)
    assert status == "ok" and rows[0].context_window == 8192
    spec = build_model_spec("vllm", "model/with:tag")
    assert spec.base_url == first and spec.context_window == 8192 and not spec.supports_tools
    assert not spec.reasoning_efforts and spec.temperature is None
    second = first.replace("/proxy/", "/another/")
    write_setting(manager, BY_KEY["providers.vllm.base_url"], second)
    assert resolve_base_url("vllm") == second
    assert cached_available_models("vllm")[0] == []
    state["models"] = [{"id": "second-server", "max_model_len": 2048}]
    rows, status = available_models("vllm", api_key=None)
    assert status == "ok" and [r.id for r in rows] == ["second-server"]
    assert any(path == "/another/v1/models" for path, _ in state["requests"])


def test_native_lmstudio_loaded_alias_and_tri_state(local_server):
    state = local_server
    state["models"] = [
        {"id": "library-key"},
        {"id": "loaded-alias"},
        {"id": "embedding"},
        {"id": "unknown"},
    ]
    state["native"] = {
        "models": [
            {
                "key": "library-key",
                "type": "llm",
                "max_context_length": 40960,
                "loaded_instances": [{"id": "loaded-alias", "config": {"context_length": 4096}}],
                "capabilities": {
                    "vision": False,
                    "trained_for_tool_use": True,
                    "reasoning": {"allowed_options": ["off", "on"], "default": "on"},
                },
            },
            {"key": "embedding", "type": "embedding"},
        ]
    }
    with httpx.Client() as client:
        rows = {
            row.id: row for row in discover_local("lmstudio", state["endpoint"], None, client, 5)
        }
    assert "embedding" not in rows
    assert rows["loaded-alias"].active_context_window == 4096
    assert rows["loaded-alias"].max_context_window == 40960
    assert rows["loaded-alias"].supports_images is False
    assert rows["loaded-alias"].supports_tools is True
    assert rows["loaded-alias"].reasoning is True
    assert rows["unknown"].supports_tools is None


@pytest.mark.asyncio
async def test_setup_persists_only_after_confirmation_and_scopes_bearer(local_server, tmp_path):
    state = local_server
    auth = AuthStore(tmp_path / "auth.db")
    answers = iter([state["endpoint"], "fixture-only-token", "model/with:tag", "yes"])
    controller = ProviderController(
        auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(answers))
    )
    receipt = await controller.login("vllm")
    assert "Configured vLLM" in receipt
    config = ConfigManager(tmp_path / "config")
    assert config.get_config_value("hosting") == "vllm"
    assert config.get_config_value("model_name") == "model/with:tag"
    assert await auth.get_api_key("vllm") == "fixture-only-token"
    write_setting(
        config, BY_KEY["providers.vllm.base_url"], state["endpoint"].replace("/proxy/", "/other/")
    )
    assert await auth.get_api_key("vllm") is None
    auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", range(4))
async def test_setup_cancellation_preserves_config(local_server, tmp_path, cancel_at):
    state = local_server
    manager = state["manager"]
    manager.update_config({"hosting": "openai", "model_name": "existing"})
    before = (tmp_path / "config" / "config.yml").read_bytes()
    answers = [state["endpoint"], "", "model/with:tag", "yes"]
    answers[cancel_at] = None
    remaining = iter(answers)
    auth = AuthStore(tmp_path / "auth.db")
    controller = ProviderController(
        auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(remaining))
    )
    with pytest.raises(LoginError, match="cancelled"):
        await controller.login("vllm")
    assert (tmp_path / "config" / "config.yml").read_bytes() == before
    assert auth.list_credentials("vllm") == []
    auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,models,message",
    [(401, [], "rejected"), (200, [], "no chat models"), (503, [], "HTTP 503")],
)
async def test_failed_probe_preserves_config(local_server, tmp_path, status, models, message):
    state = local_server
    state.update(status=status, models=models)
    before = (tmp_path / "config" / "config.yml").read_bytes()
    answers = iter([state["endpoint"], ""])
    auth = AuthStore(tmp_path / "auth.db")
    controller = ProviderController(
        auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(answers))
    )
    with pytest.raises(ValueError, match=message):
        await controller.login("vllm")
    assert (tmp_path / "config" / "config.yml").read_bytes() == before
    auth.close()


@pytest.mark.asyncio
async def test_real_stream_reasoning_replay_is_endpoint_scoped(local_server):
    endpoint = local_server["endpoint"]
    model = ModelSpec(provider="vllm", model_id="fixture-model", base_url=endpoint)
    request = ChatRequest(
        model=model, messages=[Message(role="user", content=[TextContent(text="Say LOCAL_OK")])]
    )
    async with httpx.AsyncClient() as http:
        client = OpenAICompatClient(endpoint, http_client=http)
        events = [event async for event in client.stream(request, api_key=None)]
        assert "".join(e.delta for e in events if isinstance(e, StreamTextDelta)) == "LOCAL_OK"
        end = next(e for e in events if isinstance(e, StreamEndEvent))
        assert end.provider_payload is not None
        assert end.provider_payload["native_replay"]["items"] == [
            {"reasoning_content": "Private protocol continuation"}
        ]
        history = Message(
            role="assistant",
            content=[TextContent(text="LOCAL_OK")],
            provider_payload=end.provider_payload,
        )
        second = request.model_copy(update={"messages": [*request.messages, history]})
        await anext(client.stream(second, api_key=None))
        body = local_server["requests"][-1][1]
        assert body["messages"][-1]["reasoning_content"] == "Private protocol continuation"
        other = OpenAICompatClient(endpoint.replace("/proxy/", "/other/"), http_client=http)
        await anext(other.stream(second, api_key=None))
        assert "reasoning_content" not in local_server["requests"][-1][1]["messages"][-1]


def test_ollama_metadata_uses_running_context_and_known_false():
    def respond(request):
        path = request.url.path
        if path == "/v1/models":
            body = {"data": [{"id": "running:tag"}, {"id": "unloaded:tag"}]}
        elif path == "/api/ps":
            body = {"models": [{"name": "running:tag", "context_length": 2048}]}
        elif path == "/api/show":
            body = {
                "capabilities": ["completion", "tools"],
                "model_info": {"llama.context_length": 32768},
            }
        else:
            raise AssertionError(path)
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        rows = {
            row.id: row
            for row in discover_local("ollama", "http://fixture.invalid/v1", None, client, 5)
        }
    assert rows["running:tag"].context_window == 2048
    assert rows["unloaded:tag"].context_window == 4096
    assert rows["unloaded:tag"].max_context_window == 32768
    assert rows["running:tag"].supports_tools is True
    assert rows["running:tag"].supports_images is False
    assert rows["running:tag"].reasoning is False


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["auto", "none", "required"])
async def test_ollama_tool_choice_preserves_safety(choice):
    from local_operator.providers.failover import ProviderError

    async def execute(*args, **kwargs):
        return ToolResult(tool_call_id="fixture")

    tool = AgentTool(
        name="test_tool", description="fixture", parameters={"type": "object"}, execute=execute
    )
    request = ChatRequest(
        model=ModelSpec(provider="ollama", model_id="local"),
        messages=[],
        tools=[tool],
        tool_choice=choice,
    )
    async with httpx.AsyncClient() as http:
        client = OpenAICompatClient("http://fixture.invalid/v1", http_client=http)
        if choice == "required":
            with pytest.raises(ProviderError, match="does not support required"):
                client._build_body(request)
        else:
            body = client._build_body(request)
            assert "tool_choice" not in body
            assert ("tools" in body) == (choice == "auto")
        unsupported = request.model_copy(
            update={"model": request.model.model_copy(update={"supports_tools": False})}
        )
        if choice == "required":
            with pytest.raises(ProviderError, match="does not support required"):
                client._build_body(unsupported)
        else:
            assert "tools" not in client._build_body(unsupported)


@pytest.mark.asyncio
async def test_manual_listing_fallback_and_keyless_setup(local_server, tmp_path):
    local_server["status"] = 404
    answers = iter([local_server["endpoint"], "", "manual/exact:ID", "yes"])
    auth = AuthStore(tmp_path / "auth.db")
    controller = ProviderController(
        auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(answers))
    )
    await controller.login("openai-compatible")
    assert auth.list_credentials("openai-compatible") == []
    assert ConfigManager(tmp_path / "config").get_config_value("model_name") == "manual/exact:ID"
    auth.close()


def test_desktop_provider_and_model_routes_share_local_configuration(local_server):
    from unittest.mock import MagicMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from local_operator.env import EnvConfig
    from local_operator.server.dependencies import (
        get_credential_manager,
        get_env_config,
    )
    from local_operator.server.routes.models import router

    local_server["manager"].update_config(
        {"providers": {"lmstudio": {"base_url": local_server["endpoint"]}}}
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_credential_manager] = lambda: MagicMock()
    app.dependency_overrides[get_env_config] = lambda: EnvConfig()
    with TestClient(app) as client:
        providers = client.get("/v1/models/providers")
        assert providers.status_code == 200
        assert not local_server["requests"]
        entries = providers.json()["result"]["providers"]
        assert {p["id"] for p in entries} >= set(LOCAL_PRESETS)
        assert all(not p["requiredCredentials"] for p in entries if p["id"] in LOCAL_PRESETS)
        response = client.get("/v1/models", params={"provider": "lmstudio"})
        assert response.status_code == 200, response.text
        model = response.json()["result"]["models"][0]
        assert model["id"] == "model/with:tag"
        assert model["info"]["context_window"] == 8192
        assert client.get("/v1/models", params={"provider": "missing"}).status_code == 404


@pytest.mark.asyncio
async def test_stopped_server_preserves_working_configuration(local_server, tmp_path):
    import socket

    before = (tmp_path / "config" / "config.yml").read_bytes()
    auth = AuthStore(tmp_path / "auth.db")
    # A bound but non-listening socket gives a deterministic connection refusal
    # without racing another process for a recently released port.
    with socket.socket() as stopped:
        stopped.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{stopped.getsockname()[1]}/v1"
        answers = iter([endpoint, ""])
        controller = ProviderController(
            auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(answers))
        )
        with pytest.raises(ValueError, match="Cannot reach the server"):
            await controller.login("lmstudio")
    assert (tmp_path / "config" / "config.yml").read_bytes() == before
    assert not auth.list_credentials("lmstudio")
    auth.close()


def test_yaml_model_overrides_round_trip_through_text_editor(local_server):
    from local_operator.settings_io import read_setting

    value = {"exact:model": {"context_window": 4096, "supports_tools": False}}
    manager = local_server["manager"]
    manager.update_config({"providers": {"lmstudio": {"models": value}}})
    setting = BY_KEY["providers.lmstudio.models"]
    text = read_setting(manager, setting)
    assert json.loads(text) == value
    assert validate(setting, text) is None
    write_setting(manager, setting, text)
    assert model_overrides(read_setting(manager, setting)) == value


@pytest.mark.parametrize("status", [401, 404])
def test_optional_native_metadata_failure_keeps_compatible_listing(status):
    def respond(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "known", "supports_tools": False}]})
        return httpx.Response(status, json={"error": "native extension unavailable"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        rows = discover_local("lmstudio", "http://fixture.invalid/v1", None, client, 5)
    assert [r.id for r in rows] == ["known"]
    assert rows[0].supports_tools is False


def test_llamacpp_props_active_context_wins():
    def respond(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model", "context_window": 32768}]})
        assert request.url.path == "/props"
        return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 4096}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        rows = discover_local("llamacpp", "http://fixture.invalid/v1", None, client, 5)
    assert rows[0].context_window == rows[0].active_context_window == 4096

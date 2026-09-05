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
            body = (
                b"{"
                if state.get("malformed") and self.path.endswith("/v1/models")
                else json.dumps(payload).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            state["requests"].append((self.path, body))
            state.setdefault("post_authorizations", []).append(self.headers.get("Authorization"))
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
    local_server["manager"].update_config({"providers": {"vllm": {"base_url": endpoint}}})
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
        other_endpoint = endpoint.replace("/proxy/", "/other/")
        local_server["manager"].update_config({"providers": {"vllm": {"base_url": other_endpoint}}})
        other = OpenAICompatClient(other_endpoint, http_client=http)
        second = second.model_copy(
            update={"model": model.model_copy(update={"base_url": other_endpoint})}
        )
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


@pytest.mark.parametrize("provider", ["lmstudio", "openai-compatible"])
def test_desktop_provider_and_model_routes_share_local_configuration(local_server, provider):
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
        {"providers": {provider: {"base_url": local_server["endpoint"]}}}
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
        response = client.get("/v1/models", params={"provider": provider})
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


@pytest.mark.asyncio
async def test_generic_setup_reaches_normal_catalogue_and_desktop_paths(local_server, monkeypatch):
    from local_operator.providers import controller as controller_module
    from local_operator.providers.local import local_api_key

    auth = AuthStore()
    answers = iter([local_server["endpoint"], "fixture-generic", "model/with:tag", "yes"])
    controller = ProviderController(
        auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(answers))
    )
    await controller.login("openai-compatible")
    before = len(local_server["requests"])
    rows, status = available_models("openai-compatible", api_key=local_api_key("openai-compatible"))
    assert status == "ok" and [r.id for r in rows] == ["model/with:tag"]
    assert len(local_server["requests"]) > before
    definition = get_provider_definition("openai-compatible")
    assert definition is not None
    monkeypatch.setattr(controller_module, "PROVIDER_REGISTRY", [definition])
    catalogue, statuses = await controller.live_catalogue()
    assert any(
        row.provider == "openai-compatible" and row.model_id == "model/with:tag"
        for row in catalogue
    )
    generic = next(row for row in catalogue if row.provider == "openai-compatible")
    assert generic.input_price < 0 and generic.output_price < 0
    assert rows[0].free is False
    assert statuses["openai-compatible"] in {"ok", "cached"}
    assert (
        build_model_spec("openai-compatible", "model/with:tag").base_url == local_server["endpoint"]
    )
    auth.close()


def test_activation_refreshes_shrunk_capacity_without_discarding_catalogue(local_server):
    endpoint = local_server["endpoint"]
    local_server["manager"].update_config({"providers": {"vllm": {"base_url": endpoint}}})
    local_server["models"] = [{"id": "capacity", "max_model_len": 32768}]
    first = build_model_spec("vllm", "capacity")
    assert first.context_window == 32768
    reads = len(local_server["requests"])
    assert cached_available_models("vllm")[0][0].context_window == 32768
    assert len(local_server["requests"]) == reads
    local_server["models"] = [{"id": "capacity", "max_model_len": 4096}]
    second = build_model_spec("vllm", "capacity")
    assert second.context_window == 4096
    assert len(local_server["requests"]) > reads
    assert cached_available_models("vllm")[0][0].context_window == 4096
    local_server["status"] = 503
    assert build_model_spec("vllm", "capacity").context_window <= 4096


def test_lmstudio_activation_refreshes_loaded_alias_and_missing_native_metadata(local_server):
    local_server["models"] = [{"id": "loaded-alias"}]
    allocation = {"context_length": 32768}
    local_server["native"] = {
        "models": [
            {
                "type": "llm",
                "key": "library-key",
                "max_context_length": 131072,
                "loaded_instances": [{"id": "loaded-alias", "config": allocation}],
            }
        ]
    }
    local_server["manager"].update_config(
        {"providers": {"lmstudio": {"base_url": local_server["endpoint"]}}}
    )
    assert build_model_spec("lmstudio", "loaded-alias").context_window == 32768
    allocation["context_length"] = 4096
    assert build_model_spec("lmstudio", "loaded-alias").context_window == 4096
    local_server["native"] = None
    assert build_model_spec("lmstudio", "loaded-alias").context_window == 4096


@pytest.mark.parametrize(
    "provider", ["lmstudio", "ollama", "vllm", "llamacpp", "openai-compatible"]
)
def test_training_maximum_is_not_active_capacity(local_server, provider):
    local_server["models"] = [{"id": "maximum-only", "max_context_length": 131072}]
    local_server["manager"].update_config(
        {"providers": {provider: {"base_url": local_server["endpoint"]}}}
    )
    spec = build_model_spec(provider, "maximum-only")
    assert spec.context_window == 4096
    assert spec.max_context_window == 131072
    local_server["manager"].update_config(
        {
            "providers": {
                provider: {
                    "base_url": local_server["endpoint"],
                    "models": {"maximum-only": {"context_window": 8192}},
                }
            }
        }
    )
    assert build_model_spec(provider, "maximum-only").context_window == 8192


def test_ollama_filters_authoritative_embedding_only_capability():
    def respond(request):
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": "embed"}, {"id": "both"}, {"id": "unknown"}]}
            )
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        model = json.loads(request.content)["model"]
        caps = {"embed": ["embedding"], "both": ["embedding", "completion"], "unknown": None}[model]
        return httpx.Response(200, json={"capabilities": caps})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        rows = discover_local("ollama", "http://fixture.invalid/v1", None, client, 5)
    assert [r.id for r in rows] == ["both", "unknown"]
    assert rows[1].supports_tools is None


@pytest.mark.asyncio
async def test_repeated_setup_replaces_active_token_but_keeps_history(local_server):
    from local_operator.providers.local import local_api_key

    auth = AuthStore()
    for token in ("fixture-first", "fixture-second", ""):
        answers = iter([local_server["endpoint"], token, "model/with:tag", "yes"])
        controller = ProviderController(
            auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(answers))
        )
        await controller.login("vllm")
    assert all([await auth.get_api_key("vllm") == "fixture-second" for _ in range(4)])
    assert local_api_key("vllm") == "fixture-second"
    assert len(auth.list_credentials("vllm", include_disabled=True)) == 3
    answers = iter([local_server["endpoint"], "-", "model/with:tag", "yes"])
    controller = ProviderController(
        auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(answers))
    )
    await controller.login("vllm")
    assert await auth.get_api_key("vllm") is None
    assert local_api_key("vllm") is None
    history = auth.list_credentials("vllm", include_disabled=True)
    assert len(history) == 3 and all(row.disabled_cause == "local-token-cleared" for row in history)
    auth.close()


@pytest.mark.asyncio
async def test_rejected_latest_local_token_does_not_revive_history_or_cloud_env(
    local_server, monkeypatch
):
    from local_operator.providers.local import local_api_key

    endpoint = local_server["endpoint"]
    local_server["manager"].update_config({"providers": {"vllm": {"base_url": endpoint}}})
    auth = AuthStore()
    history = [
        auth.upsert_credential(
            "vllm", {"key": key, "type": "api_key", "source": "login", "endpoint": endpoint}
        )
        for key in ("fixture-original", "fixture-replacement")
    ]
    auth.disable_credential(history[-1].id, "invalidated-token")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-cloud-only")
    assert await auth.get_api_key("vllm") is None
    assert local_api_key("vllm") is None
    assert await auth.get_api_key("openai") == "fixture-cloud-only"
    assert len(auth.list_credentials("vllm", include_disabled=True)) == 2
    auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,change",
    [("vllm", "other"), ("vllm", ""), ("vllm", "ftp://invalid"), ("openai-compatible", "")],
)
async def test_endpoint_changes_never_dispatch_a_stale_client(local_server, provider, change):
    from local_operator.providers.failover import ProviderError

    endpoint = local_server["endpoint"]
    config = local_server["manager"]
    config.update_config({"providers": {provider: {"base_url": endpoint}}})
    model = ModelSpec(provider=provider, model_id="model/with:tag", base_url=endpoint)
    new_endpoint = endpoint.replace("/proxy/", "/other/") if change == "other" else change
    config.update_config({"providers": {provider: {"base_url": new_endpoint}}})
    async with httpx.AsyncClient() as http:
        client = OpenAICompatClient(endpoint, http_client=http)
        request = ChatRequest(
            model=model, messages=[Message(role="user", content=[TextContent(text="hello")])]
        )
        with pytest.raises((ProviderError, ValueError)):
            _ = [
                event
                async for event in client.stream(request, api_key="fixture-new-endpoint-token")
            ]
    assert local_server["requests"] == []
    assert local_server.get("post_authorizations", []) == []


@pytest.mark.asyncio
async def test_selected_credential_provenance_prevents_dispatch_race(local_server):
    from local_operator.providers.auth_store import OAuthAccess
    from local_operator.providers.failover import ProviderError

    endpoint = local_server["endpoint"]
    local_server["manager"].update_config({"providers": {"vllm": {"base_url": endpoint}}})
    # Configuration may have changed back since a different endpoint's key was
    # selected. Its provenance, not another config read, prevents this race.
    access = OAuthAccess(
        access_token="fixture-other",
        credential_id=1,
        kind="api_key",
        api_endpoint=endpoint.replace("/proxy/", "/other/"),
    )
    request = ChatRequest(
        model=ModelSpec(provider="vllm", model_id="model/with:tag", base_url=endpoint), messages=[]
    )
    async with httpx.AsyncClient() as http:
        client = OpenAICompatClient(endpoint, http_client=http)
        with pytest.raises(ProviderError, match="endpoint changed"):
            _ = [
                event
                async for event in client.stream(
                    request, api_key=access.access_token, oauth_access=access
                )
            ]
    assert local_server["requests"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [False, True])
async def test_local_context_overflow_advice_matches_recovery_options(history):
    from local_operator.providers.failover import ProviderError

    messages = (
        [Message(role="assistant", content=[TextContent(text="prior turn")])] if history else []
    )
    request = ChatRequest(
        model=ModelSpec(provider="lmstudio", model_id="loaded", context_window=16384),
        messages=messages,
        context_tokens_hint=16640,
        context_tokens_hint_measured=16640,
        context_tokens_hint_model="lmstudio/loaded",
    )
    async with httpx.AsyncClient() as http:
        client = OpenAICompatClient("http://fixture.invalid/v1", http_client=http)
        with pytest.raises(ProviderError) as error:
            client._build_body(request)
    assert "16,640" in str(error.value)
    assert "loaded server context" in str(error.value)
    assert "Client context overrides do not resize the server" in str(error.value)
    assert ("Compact the conversation" in str(error.value)) == history
    assert "start a new session" not in str(error.value)


@pytest.mark.asyncio
async def test_malformed_setup_list_has_safe_actionable_error(local_server):
    local_server["malformed"] = True
    before = local_server["manager"].get_config().values.copy()
    auth = AuthStore()
    answers = iter([local_server["endpoint"], ""])
    controller = ProviderController(
        auth, login_callbacks=lambda _: LoginCallbacks(on_setup_input=lambda *_: next(answers))
    )
    with pytest.raises(ValueError, match="invalid model list.*OpenAI-compatible API"):
        await controller.login("vllm")
    assert local_server["manager"].get_config().values == before
    auth.close()


@pytest.mark.asyncio
async def test_local_model_activation_and_forwarded_selection_refresh_off_loop(
    local_server, monkeypatch
):
    import asyncio

    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory, _set_editor_line

    changed = asyncio.Event()
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    resolver_threads = []

    class Session(FakeSession):
        selected: ModelSpec | None = None

        @property
        def model(self):
            return self.selected

        @property
        def model_label(self):
            return (
                f"{self.selected.provider}/{self.selected.model_id}"
                if self.selected
                else "test/model"
            )

        def set_model(self, model, *, explicit=False):
            self.selected = model
            changed.set()

    endpoint = local_server["endpoint"]
    local_server["manager"].update_config({"providers": {"vllm": {"base_url": endpoint}}})
    local_server["models"] = [{"id": "capacity", "max_model_len": 32768}]
    auth = AuthStore()
    controller = ProviderController(auth)
    resolve = controller.resolve_model

    def record_resolution(provider, model_id):
        thread = threading.get_ident()
        resolver_threads.append(thread)
        if len(resolver_threads) == 1:
            loop.call_soon_threadsafe(started.set)
            if thread != loop_thread:
                assert release.wait(timeout=30), "test never released the capacity lookup"
        return resolve(provider, model_id)

    monkeypatch.setattr(controller, "resolve_model", record_resolution)
    session = Session()
    app = OperatorApp(lambda: _factory(session), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(100):
            await pilot.pause()
            if app._session is not None:
                break
        app._cmd_model("vllm/capacity", app._notice)
        try:
            await asyncio.wait_for(started.wait(), 30)
            assert resolver_threads[0] != loop_thread
            assert app.composer_submission_blocked()
            _set_editor_line(app._editor(), "keep this draft for the new model")
            await pilot.press("enter")
            assert app._editor().text == "keep this draft for the new model"
            assert session.model is None
        finally:
            release.set()
        await asyncio.wait_for(changed.wait(), 30)
        assert not app.composer_submission_blocked()
        assert session.model is not None
        assert session.model.context_window == 32768
        local_server["models"] = [{"id": "capacity", "max_model_len": 4096}]
        result = await app.run_slash_authoritative("model", "vllm/capacity")
        assert result["kind"] == "notice"
        assert session.model is not None
        assert session.model.context_window == 4096
        assert len(resolver_threads) == 2 and all(t != loop_thread for t in resolver_threads)
        assert session.prompts == []
    auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
async def test_slow_local_model_choice_cannot_replace_a_newer_choice(routed):
    import asyncio

    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import (
        FakeModel,
        FakeProviderController,
        FakeSession,
        _factory,
    )

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    release = threading.Event()
    applied = []

    class Providers(FakeProviderController):
        def provider(self, pid):
            return get_provider_definition(pid)

        def is_usable(self, provider):
            return True

        def resolve_model(self, provider, model_id):
            if model_id == "slow":
                loop.call_soon_threadsafe(started.set)
                try:
                    assert release.wait(timeout=30), "test never released the old lookup"
                finally:
                    loop.call_soon_threadsafe(finished.set)
            return FakeModel(provider, model_id)

    class Session(FakeSession):
        def set_model(self, model, *, explicit=False):
            applied.append(model.model_id)

    session = Session()
    app = OperatorApp(lambda: _factory(session), provider_controller=Providers())
    task = None
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(100):
            await pilot.pause()
            if app._session is not None:
                break
        if routed:
            task = asyncio.create_task(app.run_slash_authoritative("model", "vllm/slow"))
        else:
            app._cmd_model("vllm/slow", app._notice)
        try:
            await asyncio.wait_for(started.wait(), 30)
            assert app.composer_submission_blocked()
            app._cmd_model("openai/fast", app._notice)
            assert applied == ["fast"]
            assert not app.composer_submission_blocked()
        finally:
            release.set()
        await asyncio.wait_for(finished.wait(), 30)
        if task is not None:
            result = await asyncio.wait_for(task, 30)
            assert result["style"] == "warning"
        for _ in range(3):
            await pilot.pause()
        assert applied == ["fast"]
        assert not app.composer_submission_blocked()

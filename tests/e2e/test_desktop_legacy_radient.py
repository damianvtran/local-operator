"""Real legacy HTTP/CLI clients against a threaded fake Radient upstream."""

import asyncio
import json
import secrets
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from local_operator.agents import AgentEditFields
from local_operator.server.app import app
from tests.e2e.test_desktop_radient import serve

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_legacy_radient_readers_use_central_store(headless_tui_env, monkeypatch):
    from local_operator.cli import agents_delete_command
    from local_operator.providers import radient_credentials

    root = headless_tui_env
    token, key = secrets.token_hex(32), secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (root / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            calls.append(("GET", self.path))
            if not self.path.endswith("/models"):
                self.respond(404, b"{}")
                return
            authorized = self.headers.get("Authorization") == "Bearer " + key
            assert authorized
            self.respond(200, b'{"data": []}')

        def do_POST(self):
            authorized = self.headers.get("Authorization") == "Bearer " + key
            assert authorized
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append(("POST", self.path))
            if self.path.endswith("/speech"):
                if json.loads(body)["input"] == "fail":
                    self.respond(400, json.dumps({"error": key}).encode())
                else:
                    self.respond(200, b"fixture-audio", "audio/mpeg")
            else:
                assert b"application/zip" in body
                self.respond(200, b'{"id": "remote-fixture"}')

        def do_DELETE(self):
            authorized = self.headers.get("Authorization") == "Bearer " + key
            assert authorized
            calls.append(("DELETE", self.path))
            self.respond(204, b"")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    monkeypatch.setenv("RADIENT_API_BASE_URL", base)
    monkeypatch.setattr(
        radient_credentials, "canonical_radient_destination", lambda value: value == base
    )
    try:
        async with serve(app) as url, httpx.AsyncClient(base_url=url, timeout=30) as client:
            speech_body = {"input": "Hello fixture", "model": "tts-fixture", "voice": "fixture"}
            for method, path, payload in [
                ("GET", "/v1/models", None),
                ("POST", "/v1/tools/speech", speech_body),
                ("POST", "/v1/agents/fixture/upload", None),
                ("POST", "/v1/transcriptions", None),
            ]:
                response = await client.request(method, path, json=payload)
                assert response.status_code == 401
                response = await client.request(
                    method,
                    path,
                    json=payload,
                    headers={"Authorization": "Bearer " + token, "Origin": "https://evil.example"},
                )
                assert response.status_code == 403
            assert not calls
            client.headers["Authorization"] = "Bearer " + token
            await client.get("/v1/auth/status")
            app.state.desktop_auth.store.upsert_credential(
                "radient", {"type": "api_key", "source": "login", "key": key}
            )
            assert not app.state.credential_manager.get_credential("RADIENT_API_KEY")
            speech = await client.post("/v1/tools/speech", json=speech_body)
            assert speech.status_code == 200 and speech.content == b"fixture-audio"
            assert speech.headers["cache-control"] == "no-store"
            models = await client.get("/v1/models?provider=radient")
            assert models.status_code == 200
            agent = app.state.agent_registry.create_agent(
                AgentEditFields.model_validate({"name": "Legacy upload fixture"})
            )
            uploaded = await client.post(f"/v1/agents/{agent.id}/upload")
            assert uploaded.status_code == 200, uploaded.status_code
            assert ("GET", "/v1/models") in calls
            assert ("POST", "/v1/agents/upload") in calls
            assert not app.state.credential_manager.get_credential("RADIENT_API_KEY")
            # The CLI intentionally retains its old endpoint joining behavior;
            # this fixture verifies the changed credential path, not a new API.
            app.state.config_manager.set_config_value("radient_base_url", base)
            result = await asyncio.to_thread(
                agents_delete_command,
                Namespace(name=None, agent_id="remote-fixture"),
                app.state.agent_registry,
                root,
            )
            assert result == 0
            assert any(method == "DELETE" for method, _ in calls)
            failure = await client.post("/v1/tools/speech", json={**speech_body, "input": "fail"})
            safe_error = key not in failure.text
            assert safe_error
            print(
                "Legacy HTTP speech200 returned audio, catalogue200 and real ZIP upload200 "
                "used only AuthStore key; CLI delete used same resolver; no key dual-write"
            )
            print(
                "Managed legacy paths: missing token401, wrong Origin403; reflected upstream "
                "credential absent from failure response; fake upstream only"
            )
    finally:
        await asyncio.to_thread(upstream.shutdown)
        upstream.server_close()
        worker.join(timeout=5)

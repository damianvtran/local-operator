"""Music's real localhost transport/attachment path, with no paid vendor calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from local_operator.clients._http import APIError
from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import ToolContext
from local_operator.music.client import MusicClient
from local_operator.music.jobs import runner
from local_operator.music.models import GenerateMusicParams
from local_operator.tools import music_tool


def test_music_controls_reject_unrepresentable_and_preserve_exact_values():
    for values in [
        dict(prompt="piano", duration_ms=2999),
        dict(prompt="piano", duration_ms=3000, variations=3),
        dict(prompt="piano", duration_ms=3000, source_url="http://localhost/private"),
        dict(duration_ms=3000, instrumental=True, lyrics="words"),
    ]:
        with pytest.raises(ValidationError):
            GenerateMusicParams.model_validate(values)
    result = GenerateMusicParams(prompt="piano", duration_ms=3000, instrumental=False)
    assert result.body()["instrumental"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [None, 42, {}, "", "https://other.invalid/tools/audio/artifacts/a"])
async def test_download_refuses_invalid_url_before_transport(url):
    def refuse_request(request):
        pytest.fail("invalid artifact URL must never reach the transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse_request)) as transport:
        client = MusicClient(base_url="https://music.invalid", client=transport)
        with pytest.raises(APIError, match="invalid_music_artifact_origin"):
            await client.download({"url": url})


@pytest.mark.asyncio
async def test_disabled_server_refuses_before_background_registration(monkeypatch):
    class Client:
        async def request(self, action):
            assert action == "models"
            return {
                "models": [
                    {
                        "id": "radient/music",
                        "enabled": False,
                        "disabled_reason": "commercial_authorization_required",
                        "operations": {"compose": False},
                    }
                ]
            }

        async def close(self):
            pass

    monkeypatch.setattr(music_tool, "MusicClient", Client)

    async def writer(record):
        pytest.fail("disabled server must not journal or submit")

    context = ToolContext(jobs=AsyncJobManager(), journal_music_job=writer)
    result = await music_tool.execute_generate_music(
        "call", {"prompt": "piano", "duration_ms": 3000}, None, None, context
    )
    assert result.is_error
    assert result.details is not None
    assert result.details["capabilities"]["models"][0]["enabled"] is False


@pytest.mark.asyncio
async def test_real_http_partial_music_keeps_charge_identity_audio_and_restart(
    monkeypatch, tmp_path
):
    data = (Path(__file__).parents[2] / "fixtures/audio/tone.mp3").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    calls = []
    cost = {
        "v": 1,
        "amount_usd": 0.0075,
        "basis": "billed",
        "units": 3000,
        "unit": "millisecond",
        "provider": "radient",
        "model": "radient/music",
        "tool_type": "music_generation",
        "price_version": "music-fixed-requested-ms@2026-11-08",
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, self.headers.get("Idempotency-Key")))
            assert self.headers.get("Authorization") == "Bearer synthetic"
            tracks = [
                {
                    "id": "one",
                    "title": "Piano",
                    "variant_index": 0,
                    "status": "IN_QUEUE",
                    "settled": False,
                    "artifacts": [],
                },
                {
                    "id": "two",
                    "variant_index": 1,
                    "status": "COMPLETED",
                    "settled": True,
                    "error_type": "audio_failed",
                    "cost": {**cost, "amount_usd": 0, "units": 0},
                    "artifacts": [],
                },
            ]
            self.send_json(
                {
                    "request_id": "one",
                    "tracks": tracks,
                    "attribution": "Authorized fixture attribution",
                }
            )

        def do_GET(self):
            calls.append((self.path, None, None))
            if "/artifacts/" in self.path:
                assert self.headers.get("Authorization") is None
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.end_headers()
                self.wfile.write(data)
                return
            artifacts = []
            if "/result?" in self.path:
                artifacts = [
                    {
                        "id": "artifact-one",
                        "role": "full_mix",
                        "url": f"{base}/tools/audio/artifacts/artifact-one?sig=synthetic",
                        "content_type": "audio/mpeg",
                        "duration_seconds": 1,
                        "sha256": digest,
                        "size_bytes": len(data),
                    }
                ]
            self.send_json(
                {
                    "tracks": [
                        {
                            "id": "one",
                            "title": "Piano",
                            "variant_index": 0,
                            "status": "COMPLETED",
                            "settled": True,
                            "cost": cost,
                            "error_type": None,
                            "artifacts": artifacts,
                        }
                    ]
                }
            )

        def send_json(self, value):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def credential(*args):
        return SecretStr("synthetic")

    monkeypatch.setattr("local_operator.music.client.resolve_radient_credential", credential)
    monkeypatch.setattr(
        "local_operator.session.attachments.attachments_dir", lambda: tmp_path / "attachments"
    )
    records = []
    spending = []
    progress = []

    async def journal(record):
        records.append(json.loads(json.dumps(record)))

    context = SimpleNamespace(
        journal_music_job=journal,
        record_channel_spend=spending.append,
        session_id="synthetic-session",
        session_disposed=lambda: False,
    )
    record = {
        "job": "music-job",
        "state": "registered",
        "params": {"operation": "compose", "prompt": "Piano", "duration_ms": 3000, "variations": 2},
        "key": "stable-purchase",
        "artifacts": [],
    }

    def factory(**kwargs):
        return MusicClient(base_url=base)

    try:
        await runner(context, record, client_factory=factory)("watcher", None, progress.append)
        final = records[-1]
        assert final["state"] == "completed"
        assert len(final["tracks"]) == 2 and final["tracks"][1]["error_type"] == "audio_failed"
        assert len(final["artifacts"]) == 1
        assert final["artifacts"][0]["track_id"] == "one"
        assert final["artifacts"][0]["material"] == "music"
        assert "?sig=" not in json.dumps(final)
        assert sum(row.amount_micro or 0 for row in spending) == 7500
        assert spending[0].billing_basis == "billed" and spending[0].unit == "milliseconds"
        assert [call[2] for call in calls if call[0].endswith("/generate")] == ["stable-purchase"]
        # A successor with paid/settled tracks but no local registration fetches
        # missing audio without buying anything again.
        final["artifacts"] = []
        final["tracks"][0].pop("artifacts_registered")
        await runner(context, final, client_factory=factory)("successor", None, progress.append)
        assert len([call for call in calls if call[0].endswith("/generate")]) == 1
        assert len(records[-1]["artifacts"]) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_session_teardown_never_requests_upstream_cancel():
    started = asyncio.Event()
    calls = []
    records = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def request(self, action, **kwargs):
            calls.append(action)
            if action == "status":
                started.set()
                await asyncio.Event().wait()
            raise AssertionError(action)

        async def close(self):
            pass

    async def journal(record):
        records.append(record)

    context = SimpleNamespace(
        journal_music_job=journal, record_channel_spend=None, session_disposed=lambda: True
    )
    record = {
        "job": "job",
        "key": "key",
        "params": {},
        "state": "running",
        "tracks": [{"id": "paid", "settled": False}],
    }
    task = asyncio.create_task(
        runner(context, record, client_factory=Client)("watcher", None, lambda value: None)
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == ["status"]
    assert records[-1]["state"] == "interrupted"

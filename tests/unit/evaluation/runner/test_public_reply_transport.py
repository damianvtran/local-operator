"""Bounded loopback SSE proof, not a vision or model-quality benchmark.

Use the existing stdlib HTTP server pattern and the production client factory:
only the provider's answers and the adapter boundary are deterministic fixtures.
No replaced HTTP transport, second client implementation or global provider config.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from local_operator.compaction.pass_ import run_compaction_pass
from local_operator.compaction.thresholds import CompactionSettings
from local_operator.evaluation.adapters.api import observation_content_id
from local_operator.evaluation.evidence.models import (
    ContextCompactionPayload,
    ModelResponsePayload,
    RouteIdentity,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from local_operator.evaluation.runner.episode import EpisodeRunner
from local_operator.evaluation.runner.provider_client import (
    create_provider_model_client,
)
from local_operator.harness.types import ImageContent, ModelSpec
from local_operator.providers.auth_store import AuthStore
from tests.unit.evaluation.runner import conftest as fixtures
from tests.unit.evaluation.runner.test_episode import _rescue_ok
from tests.unit.evaluation.runner.test_provider_client import (
    _distinct_framed_observation,
)
from tests.unit.evaluation.runner.test_public_reply import envelope


@pytest.mark.asyncio
async def test_loopback_sse_factory_public_memory_and_secret_retention(
    tmp_path: Path,
    episode_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_fact = "Visible status: ready"
    secret = "fixture-resolved-secret-not-for-memory"
    requests: list[dict[str, Any]] = []
    replies: list[str] = []
    decisions = 0
    summaries = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            nonlocal decisions, summaries
            self.connection.settimeout(3)
            if self.path != "/v1/chat/completions" or len(requests) >= 14:
                self.send_error(400)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            rendered = json.dumps(request["messages"], ensure_ascii=False)
            if "<conversation>" in rendered:
                summaries += 1
                # Echo only a fact demonstrably present in the actual summary
                # request. This validates transport/retention, not understanding.
                reply = public_fact if public_fact in rendered else "No retained fact"
            else:
                decisions += 1
                content = request["messages"][-1]["content"]
                text = (
                    content
                    if isinstance(content, str)
                    else "\n".join(block["text"] for block in content if block["type"] == "text")
                )
                oid = next(
                    line.split(": ", 1)[1]
                    for line in text.splitlines()
                    if line.startswith("Observation ID: ")
                )
                action: dict[str, Any] = {"kind": "wait", "observation_id": oid, "duration_ms": 1}
                if decisions == 13:
                    action = {
                        "kind": "finish",
                        "observation_id": oid,
                        "status": "done",
                        "reason": "fixture complete",
                    }
                note = public_fact if decisions == 1 else secret if decisions == 2 else ""
                reply = envelope(json.dumps({"actions": [action]}), note)
            replies.append(reply)
            # Two content deltas force the real SSE decoder to assemble the
            # envelope before decision parsing; usage takes its normal route.
            chunks = [reply[: len(reply) // 2], reply[len(reply) // 2 :]]
            events = [
                {
                    "id": f"fixture-{len(requests)}",
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                for chunk in chunks
            ]
            events.append(
                {
                    "id": f"fixture-{len(requests)}",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            )
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            data = (body + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    # Test-local proxy bypass prevents ambient proxy settings routing loopback
    # traffic elsewhere; no user's provider configuration is read or changed.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    store = AuthStore(tmp_path / "fixture-auth.db")
    store.upsert_credential("openai", {"key": "fixture-local-only", "source": "login"})
    config = fixtures.build_config(tmp_path, max_steps=13)
    route = RouteIdentity(provider_id="openai", route_id="loopback", model_id="fixture-model")
    spec = replace(fixtures.build_spec(episode_id), requested_route=route)
    model = ModelSpec(
        provider="openai",
        model_id="fixture-model",
        supports_images=True,
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        context_window=128_000,
    )
    client = create_provider_model_client(
        auth_store=store,
        settings={"providers": {"openai": {"api": "chat_completions"}}, "retry": {"maxRetries": 0}},
        route=route,
        model_spec=model,
        artifact_root=config.artifact_root,
        episode_id=episode_id,
        keep_recent_frames=3,
        rebuild_every_frames=8,
        compaction=CompactionSettings(strategy="context-full", keep_recent_tokens=100),
    )

    def framed_observation(episode: str, sequence: int, **kwargs: Any) -> Any:
        original = _distinct_framed_observation(config.artifact_root, sequence, bytes([sequence]))
        updated = original.model_copy(update={"episode_id": episode, "text": None})
        return updated.model_copy(update={"observation_id": observation_content_id(updated)})

    monkeypatch.setattr(fixtures, "observation", framed_observation)
    thread.start()
    try:
        runner = EpisodeRunner(
            spec,
            config,
            selector=fixtures.selector(tmp_path),
            model=client,
            launch=lambda _: fixtures.FakeAdapter(tmp_path, episode_id),
            rescue=_rescue_ok,
            redactions=RedactionSet.from_resolved_values([secret]),
        )
        outcome = await asyncio.wait_for(runner.run(), timeout=30)
        assert outcome.status == "completed", outcome
        root = outcome.bundle_root
        assert root is not None
        report = verify_bundle(root)
        assert report.valid, [issue.code for issue in report.issues]
        assert decisions == 13 and summaries == 0
        assert secret in replies[1]  # The fixture really emitted the unsafe note.
        responses = fixtures.payloads(root, ModelResponsePayload)
        assert len(responses) == 13
        for index, expected in ((0, public_fact), (1, "[redacted public observations]")):
            ref = responses[index].redacted_response
            assert ref is not None
            recorded = (root / "artifacts" / ref.sha256).read_text()
            assert json.loads(recorded)["public_observations"] == expected
            # The next *HTTP request* contains exactly the evidence-redacted
            # accepted reply, proving the normal Message/provider wire path.
            assistant = [
                m["content"] for m in requests[index + 1]["messages"] if m["role"] == "assistant"
            ]
            assert recorded in assistant
        assert all(secret not in json.dumps(request) for request in requests)
        assert not any(secret.encode() in p.read_bytes() for p in root.rglob("*") if p.is_file())
        compactions = fixtures.payloads(root, ContextCompactionPayload)
        assert any(record.frames_dropped == 9 for record in compactions)
        history = client._context.messages
        assert sum(isinstance(block, ImageContent) for m in history for block in m.content) < 12
        assert public_fact in "\n".join(m.text for m in history)

        async def summarize(prompt: str) -> str:
            text, *_ = await client._stream(client._summary_request(prompt))
            return text

        compacted = await asyncio.wait_for(
            run_compaction_pass(
                history,
                model=model,
                settings=CompactionSettings(strategy="context-full", keep_recent_tokens=100),
                summarize=summarize,
                now_ms=10_000,
                last_activity_ms=10_000,
                respect_threshold=False,
            ),
            timeout=10,
        )
        assert compacted.ran and summaries == 1 and len(requests) == 14
        assert public_fact in compacted.messages[0].text
        assert secret not in json.dumps(requests[-1])
        assert not any(secret in m.text for m in compacted.messages)
        # Visible under pytest -s for the offline handoff: actual response and
        # side-effect evidence, without dumping bodies or any credential value.
        print(
            json.dumps(
                {
                    "transport": "loopback HTTP/SSE via create_provider_model_client",
                    "http_requests": len(requests),
                    "accepted_decisions": decisions,
                    "response_artifacts": len(responses),
                    "bundle_valid": report.valid,
                    "frames_dropped": [record.frames_dropped for record in compactions],
                    "retained_public_fact": public_fact,
                    "secret_note": "redacted before artifact and next HTTP request",
                    "subsequent_text_summary": compacted.summary_text,
                }
            )
        )
    finally:
        await client._stream_fn.close()
        store.close()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()

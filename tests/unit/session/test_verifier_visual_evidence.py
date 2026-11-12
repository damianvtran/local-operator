"""Typed verification exercises the real SDK, child and provider wire offline."""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from local_operator.harness.types import ChatRequest, ImageContent, Message, TextContent
from local_operator.harness.verifier_evidence import (
    VerifierEvidence,
    VerifierEvidenceItem,
    assess_completion,
)
from local_operator.providers.clients import OpenAICompatClient
from tests.unit.session.test_launch_subagent import MODEL
from tests.unit.session.test_verify_completion import (
    TreeStream,
)
from tests.unit.session.test_verify_completion import config_dir as _config_dir
from tests.unit.session.test_verify_completion import make_parent, text_turn, tool_turn

config_dir = _config_dir


def image() -> ImageContent:
    out = io.BytesIO()
    Image.new("RGB", (13, 7), (19, 107, 231)).save(out, format="PNG")
    return ImageContent(data=base64.b64encode(out.getvalue()).decode())


def evidence() -> VerifierEvidence:
    return VerifierEvidence(
        items=(
            VerifierEvidenceItem("capture-1", image()),
            VerifierEvidenceItem("text-1", TextContent(text="a worker claim")),
        ),
        required_modalities=frozenset({"image"}),
    )


def answer(status="MET", modality="image", refs=None) -> str:
    return json.dumps(
        {
            "requirements": [
                {
                    "requirement": "The deliverable has the requested appearance",
                    "status": status,
                    "modality": modality,
                    "evidence_ids": ["capture-1"] if refs is None else refs,
                    "findings": "Observed the supplied capture",
                }
            ]
        }
    )


@pytest.mark.parametrize(
    "response,expected",
    [
        (answer(), "PASS"),
        (answer("UNMET"), "FAIL"),
        (answer("UNVERIFIED"), "INCONCLUSIVE"),
        (answer(refs=["text-1"]), "INCONCLUSIVE"),
        (answer(refs=["unavailable"]), "INCONCLUSIVE"),
        (answer(modality="text", refs=["text-1"]), "INCONCLUSIVE"),
        (answer() + "\nVERDICT: PASS", "INCONCLUSIVE"),
        ('{"requirements":[]}', "INCONCLUSIVE"),
        ("Cannot see the screen\nVERDICT: PASS", "INCONCLUSIVE"),
    ],
)
def test_typed_assessment_never_parses_a_prose_pass(response, expected):
    assert assess_completion(response, evidence())[0] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_image", [False, True])
async def test_sdk_child_http_request_carries_distinctive_image(tmp_path, config_dir, tool_image):
    captured = []

    def handler(wire):
        captured.append(json.loads(wire.content))
        chunk = {"choices": [{"index": 0, "delta": {"content": answer()}, "finish_reason": None}]}
        end = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        if tool_image and len(captured) == 1:
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "synthetic-artifact",
                                    "type": "function",
                                    "function": {
                                        "name": "inspect_command",
                                        "arguments": json.dumps({"artifact_id": "capture-1"}),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            end = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        return httpx.Response(
            200,
            text=f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(end)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatClient("https://synthetic.invalid/v1", http_client=http)

        class WireTree(TreeStream):
            def fork(self, _name):
                return WireTree(self.reply, root=self.root)

            def __call__(self, request, signal):
                self.root.requests.append(request)

                async def generate():
                    try:
                        async for event in client.stream(request, "synthetic-key"):
                            yield event
                    finally:
                        row = SimpleNamespace(
                            request_id="synthetic", job_id=self.job_id, actor_role=self.actor_role
                        )
                        for callback in list(self.root.observers):
                            callback(row)

                return generate()

        stream = WireTree(lambda _: [])
        parent = make_parent(tmp_path, stream)
        from local_operator.evaluation.completion_verifier import VerifierInspector

        async def forbidden(*_args):
            raise AssertionError("artifact mode performed IO")

        inspector = VerifierInspector(run_command=forbidden, window=lambda: 1)
        inspector.evidence = evidence()
        result = await parent.verify_completion(
            request="Check the appearance",
            claim="done",
            typed_evidence=evidence(),
            host_tools=[inspector.tool()] if tool_image else [],
        )
        assert result.status == "PASS", result
        blocks = [
            block
            for msg in captured[0]["messages"]
            for block in (msg.get("content") or [])
            if isinstance(block, dict)
        ]
        [wire_image] = [block for block in blocks if block.get("type") == "image_url"]
        assert wire_image["image_url"]["url"] == "data:image/png;base64," + image().data
        assert stream.requests[0].required_image_count == 1
        if tool_image:
            assert len(captured) == 2
            images = [
                block
                for message in captured[1]["messages"]
                for block in message.get("content", [])
                if isinstance(block, dict) and block.get("type") == "image_url"
            ]
            assert len(images) == 2, captured[1]
            assert all(
                block["image_url"]["url"] == wire_image["image_url"]["url"] for block in images
            )
            assert stream.requests[1].required_image_count == 2
        await parent.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("supports,stated", [(False, True), (True, False)])
async def test_required_media_refuses_text_only_or_unknown_model(
    tmp_path, config_dir, supports, stated
):
    stream = TreeStream(lambda _: text_turn(answer()))
    parent = make_parent(tmp_path, stream)
    parent._model = parent.model.model_copy(
        update={"supports_images": supports, "supports_images_stated": stated}
    )
    result = await parent.verify_completion(
        request="Check", claim="done", typed_evidence=evidence()
    )
    assert result.status == "INCONCLUSIVE"
    assert not stream.requests
    await parent.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("data", ["", "broken-base64", base64.b64encode(b"not an image").decode()])
async def test_required_media_never_passes_missing_or_corrupt_image(tmp_path, config_dir, data):
    stream = TreeStream(lambda _: text_turn(answer()))
    parent = make_parent(tmp_path, stream)
    typed = VerifierEvidence(
        (VerifierEvidenceItem("capture-1", ImageContent(data=data)),), frozenset({"image"})
    )
    result = await parent.verify_completion(request="Check", claim="done", typed_evidence=typed)
    assert result.status == "INCONCLUSIVE", result
    await parent.dispose()


def test_actual_serving_model_and_degradation_are_checked():
    request = ChatRequest(
        model=MODEL, messages=[Message.user("image", images=[image()])], required_image_count=1
    )
    request.validate_required_images()
    with pytest.raises(ValueError, match="required image"):
        request.with_model(
            request.model.model_copy(update={"supports_images": False})
        ).validate_required_images()
    with pytest.raises(ValueError, match="required image"):
        request.model_copy(
            update={"messages": [Message.user("image omitted")]}
        ).validate_required_images()


@pytest.mark.asyncio
async def test_artifact_id_returns_image_in_child_tool_content_and_counts_budget(
    tmp_path, config_dir
):
    from local_operator.evaluation.completion_verifier import VerifierInspector
    from local_operator.harness.verifier import VerifierLimits

    async def forbidden_command(*_args):
        raise AssertionError("artifact mode must not read a controller or guest path")

    inspector = VerifierInspector(run_command=forbidden_command, window=lambda: 1)
    inspector.evidence = evidence()
    seen = []

    def reply(request):
        results = [m for m in request.messages if m.role == "tool"]
        if not results:
            return tool_turn("inspect_command", {"artifact_id": "capture-1"})
        seen.append(results[-1])
        if len(results) == 1:
            return tool_turn("inspect_command", {"artifact_id": "capture-1"})
        return text_turn(answer())

    parent = make_parent(tmp_path, TreeStream(reply))
    result = await parent.verify_completion(
        request="Check",
        claim="done",
        typed_evidence=evidence(),
        host_tools=[inspector.tool()],
        limits=VerifierLimits(max_commands=1),
    )
    assert result.status == "PASS", result
    assert result.commands == 1 and "commands" in result.caps_hit
    assert any(
        isinstance(block, ImageContent) and block.data == image().data for block in seen[0].content
    )
    assert not any(isinstance(block, ImageContent) for block in seen[-1].content)
    assert "inspect_command" not in {tool.name for tool in parent._tools}
    await parent.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"artifact_id": "/etc/passwd"},
        {"artifact_id": "https://host/image"},
        {"artifact_id": "capture-1", "command": "cat /etc/passwd"},
        {"artifact_id": "capture-1", "timeout_s": 1},
    ],
)
async def test_artifact_mode_denies_unregistered_paths_and_mixed_arguments(arguments):
    from local_operator.evaluation.completion_verifier import VerifierInspector

    async def forbidden(*_args):
        raise AssertionError("arbitrary paths must never reach an IO function")

    inspector = VerifierInspector(run_command=forbidden, window=lambda: 1)
    inspector.evidence = evidence()
    result = await inspector.tool().execute("synthetic-call", arguments)
    assert result.is_error and not any(isinstance(block, ImageContent) for block in result.content)


@pytest.mark.asyncio
async def test_required_image_rejection_never_becomes_text_only_pass(tmp_path, config_dir):
    from local_operator.harness.types import StreamEndEvent

    stream = TreeStream(
        lambda _: [StreamEndEvent(stop_reason="error", error="could not process image")]
    )
    parent = make_parent(tmp_path, stream)
    result = await parent.verify_completion(
        request="Check", claim="done", typed_evidence=evidence()
    )
    assert result.status == "INCONCLUSIVE", result
    assert all(
        any(isinstance(b, ImageContent) for m in r.messages for b in m.content)
        for r in stream.requests
    )
    await parent.dispose()


@pytest.mark.asyncio
async def test_capped_typed_unmet_assessment_keeps_repair_challenge(
    tmp_path, config_dir, monkeypatch
):
    import asyncio

    from local_operator.harness import subagent
    from local_operator.harness.verifier import VerifierLimits, build_verdict_challenge

    finish = subagent._finish_child_browser

    async def delayed(child, comms, job_id, status, *, parent_session=None):
        if status == "completed":
            await asyncio.sleep(30)
        return await finish(child, comms, job_id, status, parent_session=parent_session)

    monkeypatch.setattr(subagent, "_finish_child_browser", delayed)
    parent = make_parent(tmp_path, TreeStream(lambda _: text_turn(answer("UNMET"))))
    result = await parent.verify_completion(
        request="Check", claim="done", typed_evidence=evidence(), limits=VerifierLimits(wall_s=1)
    )
    assert result.status == "FAIL" and "wall" in result.caps_hit, result
    assert "UNMET:" in build_verdict_challenge(result.status, result.findings)
    await parent.dispose()


@pytest.mark.asyncio
async def test_actual_failover_dispatch_does_not_send_required_media_to_text_model():
    from local_operator.providers.failover import stream_with_failover
    from tests.unit.providers.test_failover import FakeAuth, ScriptedClient

    client = ScriptedClient([])

    async def client_for(_spec):
        return client

    request = ChatRequest(
        model=MODEL.model_copy(update={"supports_images": False}),
        messages=[Message.user("capture", images=[image()])],
        required_image_count=1,
    )
    with pytest.raises(ValueError, match="required image"):
        _ = [
            event
            async for event in stream_with_failover(
                request, FakeAuth({MODEL.provider: ["synthetic-key"]}), None, client_for
            )
        ]
    assert client.calls == 0


@pytest.mark.asyncio
async def test_text_inspection_receipts_can_be_cited_but_do_not_leak_across_checks(
    tmp_path, config_dir
):
    from tests.unit.session.test_verify_completion import inspect_tool

    typed = VerifierEvidence()

    def reply(request):
        if not any(m.role == "tool" for m in request.messages):
            return tool_turn("inspect_command", {"command": "read-deliverable"})
        return text_turn(answer(modality="text", refs=["inspection-1"]))

    parent = make_parent(tmp_path, TreeStream(reply))
    result = await parent.verify_completion(
        request="Check", claim="done", typed_evidence=typed, host_tools=[inspect_tool([])]
    )
    assert result.status == "PASS", result
    assert typed.registry == {}, "the reusable envelope retained an earlier check's evidence"
    await parent.dispose()


@pytest.mark.asyncio
async def test_required_images_that_do_not_fit_are_not_replaced_with_text(
    tmp_path, config_dir, monkeypatch
):
    monkeypatch.setattr(
        "local_operator.session.session.Session._wire_bytes_budget", lambda _self: 1
    )
    stream = TreeStream(lambda _: text_turn(answer()))
    parent = make_parent(tmp_path, stream)
    result = await parent.verify_completion(
        request="Check", claim="done", typed_evidence=evidence()
    )
    assert result.status == "INCONCLUSIVE", result
    assert not stream.requests
    await parent.dispose()


@pytest.mark.asyncio
async def test_legacy_verifier_does_not_run_typed_decoder_or_required_media_validation(
    tmp_path, config_dir, monkeypatch
):
    def forbidden(*_args):
        raise AssertionError("ordinary caller entered typed evidence path")

    monkeypatch.setattr("local_operator.harness.verifier_evidence.assess_completion", forbidden)
    monkeypatch.setattr(
        "local_operator.harness.verifier_evidence.validate_required_image", forbidden
    )
    parent = make_parent(tmp_path, TreeStream(lambda _: text_turn("VERDICT: PASS")))
    result = await parent.verify_completion(request="Check", claim="done")
    assert result.status == "PASS", result
    await parent.dispose()

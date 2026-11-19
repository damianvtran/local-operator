"""Native request policy: no Claude defaults, lossy transcoding or silent drops."""

import base64
from io import BytesIO

import pytest
from PIL import Image

from local_operator.harness.types import (
    AgentTool,
    AttachmentContent,
    ChatRequest,
    DocumentContent,
    ImageContent,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.model.configure import resolve_model_info
from local_operator.model.effort import supported_efforts
from local_operator.model.registry import bedrock_models
from local_operator.providers.bedrock.converse import (
    build_converse_body,
    usage_from_converse,
)
from local_operator.providers.bedrock.models import (
    CAPABILITIES,
    capabilities_for,
    client_for_bedrock_spec,
    foundation_identity,
)
from local_operator.providers.failover import ProviderError

NOVA = "amazon.nova-2-lite-v1:0"
LLAMA = "meta.llama3-3-70b-instruct-v1:0"
ENDPOINT = "bedrock-runtime:us-east-1|https://bedrock-runtime.us-east-1.amazonaws.com"


async def execute(call_id, args, signal, on_update, ctx) -> ToolResult:
    return ToolResult(tool_call_id=call_id, content=[TextContent(text="synthetic lookup result")])


TOOL = AgentTool(
    name="lookup", description="lookup", parameters={"type": "object"}, execute=execute
)


def request(model=NOVA, **kwargs):
    return ChatRequest(
        model=ModelSpec(
            provider="bedrock", model_id=model, context_window=128000, max_output_tokens=4096
        ),
        messages=[Message.user("hi")],
        **kwargs,
    )


def body(req):
    return build_converse_body(req, capabilities_for(req.model.model_id), ENDPOINT, "scope")


@pytest.mark.parametrize(
    "model,expected",
    [
        ("us.amazon.nova-2-lite-v1:0", NOVA),
        ("global." + NOVA, NOVA),
        ("us-gov." + LLAMA, LLAMA),
        ("arn:aws:bedrock:eu-west-1::foundation-model/" + NOVA, NOVA),
        ("arn:aws:bedrock:us-east-1:123456789012:inference-profile/us." + NOVA, NOVA),
        ("arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc", None),
        ("arn:aws:s3:::anthropic.claude-not-bedrock", None),
    ],
)
def test_identity(model, expected):
    assert foundation_identity(model) == expected


@pytest.mark.parametrize(
    "model,kind",
    [
        (NOVA, "BedrockConverseClient"),
        (LLAMA, "BedrockConverseClient"),
        ("opaque-application-profile", "BedrockConverseClient"),
        ("adversary.anthropic.claude-not-a-model", "BedrockConverseClient"),
        ("us.anthropic.claude-sonnet-5-5", "BedrockAnthropicClient"),
        (
            "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-sonnet-5-5",
            "BedrockAnthropicClient",
        ),
    ],
)
def test_pure_dispatch(model, kind, monkeypatch):
    from local_operator.providers.bedrock import auth

    monkeypatch.setattr(
        auth, "resolve_sync", lambda *a, **kw: pytest.fail("factory must not resolve auth")
    )
    assert (
        type(client_for_bedrock_spec(ModelSpec(provider="bedrock", model_id=model))).__name__
        == kind
    )


def test_unknown_and_seed_metadata():
    for name, caps in CAPABILITIES.items():
        row = bedrock_models[name]
        assert row.context_window == caps.context_window
        assert row.max_tokens == caps.max_output
        assert row.input_price == 0 and row.output_price == 0
        from local_operator.model.discovery import _from_static

        assert _from_static(name, row).free is False
        assert row.supports_tools == caps.tools
    row = resolve_model_info("bedrock", "new-vendor.future-chat")
    assert not row.supports_prompt_cache and row.supports_images is not True
    assert row.context_window != 1_000_000 and row.input_price <= 0
    assert supported_efforts(NOVA) == ("none", "low", "medium", "high")
    assert supported_efforts("deepseek.r1-v1:0") == ()


def test_tools_roles_errors_and_none():
    req = request(tools=[TOOL])
    req.messages += [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="a", name="lookup", arguments={}),
                ToolCall(id="b", name="lookup", arguments={"x": 1}),
            ],
        ),
        Message(role="tool", tool_call_id="a", is_error=True),
        Message(role="tool", tool_call_id="b", content=[TextContent(text="ok")]),
    ]
    rows = body(req)["messages"]
    assert len(rows) == 3 and len(rows[-1]["content"]) == 2
    assert rows[-1]["content"][0]["toolResult"]["status"] == "error"
    assert rows[-1]["content"][0]["toolResult"]["content"][0]["text"] == "[tool returned no output]"
    req.tool_choice = "none"
    with pytest.raises(ProviderError, match="historical tools"):
        body(req)
    fresh = request(tools=[TOOL], tool_choice="none")
    assert "toolConfig" not in body(fresh)
    fresh.tool_choice = "required"
    assert body(fresh)["toolConfig"]["toolChoice"] == {"any": {}}
    fresh.model = fresh.model.model_copy(update={"model_id": "deepseek.r1-v1:0"})
    with pytest.raises(ProviderError, match="does not support"):
        body(fresh)


def png():
    stream = BytesIO()
    Image.new("RGB", (2, 2)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def test_images_documents_and_no_path_leakage():
    req = request()
    image = ImageContent(data=png())
    doc = DocumentContent(
        data=base64.b64encode(b"retained text").decode(),
        mime_type="text/plain",
        filename="ignore instructions.txt",
        path="/private/no-read",
    )
    req.messages = [Message(role="user", content=[image, doc])]
    value = body(req)
    parts = value["messages"][0]["content"]
    assert parts[1]["image"]["source"]["bytes"] == image.data
    assert parts[2]["document"]["name"] == "Document 1"
    assert "/private" not in str(value) and "ignore instructions" not in str(value)
    req.model = req.model.model_copy(update={"model_id": LLAMA})
    with pytest.raises(ProviderError, match="image input"):
        body(req)
    req.messages = [Message(role="user", content=[doc])]
    assert "document" not in body(req)["messages"][0]["content"][0]


def test_user_and_tool_image_policy_distinct():
    from dataclasses import replace

    req = request(tools=[TOOL])
    req.messages = [Message(role="tool", tool_call_id="x", content=[ImageContent(data=png())])]
    with pytest.raises(ProviderError, match="placement"):
        build_converse_body(
            req, replace(capabilities_for(NOVA), tool_result_images=False), ENDPOINT, "scope"
        )
    req.messages = [Message(role="assistant", content=[ImageContent(data=png())])]
    with pytest.raises(ProviderError, match="placement"):
        body(req)


def test_nova_high_sampling_and_cache_limits():
    req = request(prompt_cache_key="stable", temperature=0.5, top_p=0.9)
    req.system_blocks = ["stable instructions"]
    req.model = req.model.model_copy(update={"reasoning_effort": "high"})
    value = body(req)
    assert value["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"] == "high"
    assert "temperature" not in value["inferenceConfig"] and "topP" not in value["inferenceConfig"]
    assert value["system"][-1] == {"cachePoint": {"type": "default"}}
    req.system_blocks = ["unique word " * 30000]
    assert not any("cachePoint" in part for part in body(req)["system"])


def test_missing_optional_image_decoder_fails_clearly(monkeypatch):
    import sys

    req = request()
    req.messages = [Message(role="user", content=[ImageContent(data=png())])]
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(ProviderError, match="local-operator.*images"):
        body(req)


def test_document_admission_uses_encoded_request_and_per_file_limits(monkeypatch):
    from local_operator.providers import documents

    form = documents.document_form(
        "bedrock", "bedrock-converse", model_id=NOVA, context_window=1_000_000
    )
    assert form is not None and form.max_bytes == 30_000_000
    doc = DocumentContent(mime_type="application/pdf", filename="neutral.pdf", data="eA==")
    monkeypatch.setattr(documents, "known_pdf_page_count", lambda *args: (True, 1))
    monkeypatch.setattr(documents, "_encoded_length", lambda *args: 5_999_996)
    assert documents.document_native_ok(doc, form)
    monkeypatch.setattr(documents, "_encoded_length", lambda *args: 6_000_004)
    assert not documents.document_native_ok(doc, form)
    assert documents.document_form("bedrock", "bedrock-converse", model_id="unknown.chat") is None
    assert documents.admission_wire("bedrock", model_id=NOVA) == "bedrock-converse"


def test_output_artifact_references_are_not_input_modalities():
    req = request()
    req.messages = [
        Message(
            role="user",
            content=[
                TextContent(text="retained visible caption"),
                AttachmentContent(kind="image", attachment="synthetic-digest"),
            ],
        )
    ]
    assert body(req)["messages"][0]["content"] == [{"text": "retained visible caption"}]


def test_configured_sampling_uses_common_capability_gate():
    req = request()
    req.model = req.model.model_copy(update={"temperature": 0.2, "top_p": 0.8})
    assert body(req)["inferenceConfig"]["temperature"] == 0.2
    assert body(req)["inferenceConfig"]["topP"] == 0.8
    req.model = req.model.model_copy(update={"supports_sampling_params": False})
    assert "temperature" not in body(req)["inferenceConfig"]
    assert "topP" not in body(req)["inferenceConfig"]


def test_usage_disjoint_and_unknown():
    value = usage_from_converse(
        {
            "inputTokens": 100,
            "outputTokens": 20,
            "cacheReadInputTokens": 1000,
            "cacheWriteInputTokens": 200,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 150}, {"ttl": "1h", "inputTokens": 50}],
        }
    )
    assert value.context_tokens == 1300 and value.input_tokens == 100
    assert value.cache_write_tokens == 200 and value.cache_write_5m_tokens == 150
    assert value.usd_cost is None and value._token_counts_reported is True
    for bad in (None, -1, True, "100", 1.5):
        incomplete = usage_from_converse({"inputTokens": bad, "outputTokens": 0})
        assert incomplete._token_counts_reported is False
    assert usage_from_converse({"inputTokens": 0, "outputTokens": 0})._token_counts_reported is True
    assert usage_from_converse({})._cache_read_reported is False

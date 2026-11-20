"""All-vendor discovery, partial-inventory authority and persisted profile facts."""

import dataclasses
import json
from types import SimpleNamespace

import httpx
import pytest

from local_operator.harness.types import ModelSpec
from local_operator.model.discovery import _listing_replaces_static
from local_operator.providers.bedrock import discovery
from local_operator.providers.bedrock.models import (
    capabilities_for_spec,
    client_for_bedrock_spec,
)

from .test_converse_body import LLAMA, NOVA
from .test_discovery_check import _Ctx


@pytest.fixture(autouse=True)
def bearer(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-listing-token")
    monkeypatch.delenv("LOP_BEDROCK_ENDPOINT_URL", raising=False)


def summary(model, *, outputs=None, inputs=None, types=None):
    return {
        "modelId": model,
        "modelName": model,
        "outputModalities": outputs or ["TEXT"],
        "inputModalities": inputs or ["TEXT"],
        "inferenceTypesSupported": types or ["ON_DEMAND"],
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "ACTIVE"},
    }


def test_all_vendor_pagination_backing_join_and_application_arn():
    calls = []
    app = "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/opaque"

    def handler(req):
        calls.append(req)
        if req.url.path.endswith("foundation-models"):
            return httpx.Response(
                200,
                json={
                    "modelSummaries": [
                        summary(NOVA, inputs=["TEXT", "IMAGE"], types=["INFERENCE_PROFILE"]),
                        summary(LLAMA),
                        summary("newvendor.chat"),
                        summary("amazon.titan-embed-text-v2:0", outputs=["EMBEDDING"]),
                    ]
                },
            )
        assert "typeEquals" not in req.url.params
        if "nextToken" not in req.url.params:
            return httpx.Response(
                200,
                json={
                    "inferenceProfileSummaries": [
                        {
                            "inferenceProfileId": "eu." + NOVA,
                            "status": "ACTIVE",
                            "models": [
                                {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/" + NOVA}
                            ],
                        }
                    ],
                    "nextToken": "page2",
                },
            )
        return httpx.Response(
            200,
            json={
                "inferenceProfileSummaries": [
                    {
                        "inferenceProfileId": "opaque",
                        "inferenceProfileArn": app,
                        "type": "APPLICATION",
                        "status": "ACTIVE",
                        "models": [
                            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/" + NOVA}
                        ],
                    }
                ]
            },
        )

    rows = discovery.fetch(_Ctx(handler))
    assert rows is not None
    assert [row.id for row in rows] == ["eu." + NOVA, app, LLAMA, "newvendor.chat"]
    assert len(calls) == 3 and "byProvider" not in calls[-1].url.params
    assert calls[-1].url.params["byOutputModality"] == "TEXT"
    assert _listing_replaces_static("bedrock", rows, account_scoped=False)
    # Listing facts survive the same JSON-friendly dataclass shape used by the
    # cache writer and remain sufficient for pure factory/runtime projection.
    assert rows[1].reasoning_efforts == ("none", "low", "medium", "high")
    stored = json.loads(json.dumps(dataclasses.asdict(rows[1])))
    spec = ModelSpec(provider="bedrock", model_id=app, bedrock_metadata=stored["bedrock_metadata"])
    caps = capabilities_for_spec(spec)
    assert caps.images is True and caps.tools is True and caps.reasoning_config == "nova2"
    assert type(client_for_bedrock_spec(spec)).__name__ == "BedrockConverseClient"


@pytest.mark.parametrize("denied", ["foundation-models", "inference-profiles"])
def test_partial_inventory_retains_failed_leg_static_fallback(denied):
    def handler(req):
        if req.url.path.endswith(denied):
            return httpx.Response(403, json={"message": "invoke only"})
        if req.url.path.endswith("foundation-models"):
            return httpx.Response(200, json={"modelSummaries": [summary(LLAMA)]})
        return httpx.Response(
            200,
            json={
                "inferenceProfileSummaries": [
                    {"inferenceProfileId": "eu." + NOVA, "status": "ACTIVE"}
                ]
            },
        )

    rows = discovery.fetch(_Ctx(handler))
    assert rows is not None
    assert rows and not _listing_replaces_static("bedrock", rows, account_scoped=True)


def test_repeated_pagination_token_keeps_partial_not_deletion_authority():
    count = []

    def handler(req):
        count.append(req)
        if req.url.path.endswith("foundation-models"):
            return httpx.Response(200, json={"modelSummaries": []})
        return httpx.Response(
            200,
            json={
                "inferenceProfileSummaries": [
                    {"inferenceProfileId": "eu." + NOVA, "status": "ACTIVE"}
                ],
                "nextToken": "same",
            },
        )

    rows = discovery.fetch(_Ctx(handler))
    assert rows is not None
    assert len(count) == 3 and len(rows) == 1
    assert not _listing_replaces_static("bedrock", rows, account_scoped=False)


def test_complete_empty_is_distinct_from_permission_denied():
    def empty(req):
        return httpx.Response(
            200,
            json=(
                {"modelSummaries": []}
                if req.url.path.endswith("foundation-models")
                else {"inferenceProfileSummaries": []}
            ),
        )

    assert discovery.fetch(_Ctx(empty)) == []
    assert _listing_replaces_static("bedrock", [], account_scoped=False)
    assert discovery.fetch(_Ctx(lambda _: httpx.Response(403, json={}))) is None


def test_profile_capabilities_intersect_and_do_not_infer_tool_result_images():
    caps = capabilities_for_spec(
        SimpleNamespace(model_id="opaque", bedrock_metadata={"foundation_ids": [NOVA, LLAMA]})
    )
    assert caps.tools is True and caps.images is False
    assert caps.cache_locations == frozenset() and caps.streaming_tools is False
    assert caps.tool_result_images is not True
    unknown = capabilities_for_spec(
        SimpleNamespace(model_id="opaque", bedrock_metadata={"images": True, "streaming": True})
    )
    assert unknown.images is True and unknown.tool_result_images is not True


def test_key_check_serializes_type_filter_as_http_type():
    from .test_discovery_check import _check

    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"inferenceProfileSummaries": []})

    assert _check(handler).valid is True
    assert seen[0].url.params["type"] == "SYSTEM_DEFINED"
    assert "typeEquals" not in seen[0].url.params

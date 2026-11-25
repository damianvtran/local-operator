"""The real recommendation cascade and adapters, with synthetic auth and HTTP."""

import asyncio
import json

import httpx
import pytest

from local_operator.classification.advisory import GOOGLE_VERTEX, OPENAI_SUBSCRIPTION
from local_operator.classification.recommend import RecommendationRequest
from local_operator.classification.service import ClassificationService
from local_operator.classification.types import Answer, Question
from local_operator.classification.vendors import build_vendor
from local_operator.providers.auth_store import OAuthAccess
from tests.unit.classification.support import candidate, store_row

pytestmark = pytest.mark.asyncio

OPENAI = OAuthAccess(access_token="fake-oauth", credential_id=3, org_id="fake-org", kind="oauth")
VERTEX = OAuthAccess(
    access_token="fake-vertex",
    credential_id=9,
    kind="oauth",
    raw={"project_id": "synthetic-project", "location": "us-central1"},
)
ROSTER = [candidate("orchard", description="Identify orchard fruit")]
REQUEST = RecommendationRequest(user_message="identify this apple", context=None, candidates=ROSTER)


class Auth:
    def __init__(self, **access):
        self.access = access
        self.calls = []

    async def get_oauth_access(self, provider, session_id, *, read_only, model_id):
        self.calls.append((provider, session_id, read_only, model_id))
        return self.access.get(provider)


def codex(choice="orchard", *, part_type="output_text", completed=True):
    text = json.dumps({"recommend_skill": choice})
    part = {"type": part_type, "text": text}
    event = {
        "type": "response.completed" if completed else "response.output_text.delta",
        "response": {
            "status": "completed",
            "output": [{"type": "message", "content": [part]}],
            "usage": {"input_tokens": 23, "output_tokens": 5},
        },
    }
    return httpx.Response(
        200,
        text=f"event: {event['type']}\ndata: {json.dumps(event)}\n\n",
        headers={"content-type": "text/event-stream"},
    )


def vertex(choice="orchard"):
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": json.dumps({"recommend_skill": choice})}]},
                }
            ],
            "usageMetadata": {"promptTokenCount": 31, "candidatesTokenCount": 7},
        },
    )


def native():
    return httpx.Response(
        200,
        json={
            "answers": {
                "recommend_skill": {
                    "choice": "orchard",
                    "confidence": 0.91,
                    "probabilities": {"orchard": 0.91},
                }
            }
        },
    )


def service(root, auth, transport, *, preferred="openai", **settings):
    subject = ClassificationService(config_dir=root, settings={"classification": settings})
    subject.bind_advisory_route(auth, "synthetic-service", lambda: preferred)
    subject._http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return subject


async def test_openai_uses_only_the_chatgpt_subscription_wire_and_no_fabricated_metadata(
    bare_manager,
):
    auth = Auth(openai=OPENAI)

    def transport(http):
        assert http.url == "https://chatgpt.com/backend-api/codex/responses"
        assert http.headers["authorization"] == "Bearer fake-oauth"
        assert http.headers["chatgpt-account-id"] == "fake-org"
        body = json.loads(http.content)
        assert body["model"] == "gpt-6-luna"
        assert body["stream"] is True and body["store"] is False
        assert body["reasoning"] == {"effort": "none"}
        schema = body["text"]["format"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["recommend_skill"]["enum"] == ["orchard", "none"]
        return codex()

    subject = service(bare_manager, auth, transport)
    try:
        outgoing = subject._plan_for(REQUEST)
        leg = build_vendor(
            OPENAI_SUBSCRIPTION,
            bare_manager,
            client=subject._http,
            auth_store=auth,
            session_id="synthetic-service",
        )
        from local_operator.classification.types import DecisionRequest

        response = await leg.decide(
            DecisionRequest("synthetic fruit", outgoing.questions), timeout_s=1
        )
        assert response.answers["recommend_skill"] == Answer("recommend_skill", "choice", "orchard")
        assert response.cost_usd is None
        rec = await subject.recommend_resources(REQUEST)
        assert rec.resources == tuple(ROSTER)
        assert (rec.vendor, rec.input_tokens, rec.output_tokens, rec.cost_usd) == (
            OPENAI_SUBSCRIPTION,
            23,
            5,
            None,
        )
        assert all(row == ("openai", "synthetic-service", True, "gpt-6-luna") for row in auth.calls)
    finally:
        await subject.aclose()


async def test_vertex_preference_uses_existing_project_route_and_honors_none(bare_manager):
    auth = Auth(openai=OPENAI, **{GOOGLE_VERTEX: VERTEX})

    def transport(http):
        assert http.url == (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/"
            "synthetic-project/locations/us-central1/publishers/google/models/"
            "gemini-3.1-flash-lite:generateContent"
        )
        assert http.headers["x-goog-user-project"] == "synthetic-project"
        assert http.headers["authorization"] == "Bearer fake-vertex"
        assert "x-goog-api-key" not in http.headers
        body = json.loads(http.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return vertex("none")

    subject = service(bare_manager, auth, transport, preferred=GOOGLE_VERTEX)
    try:
        rec = await subject.recommend_resources(REQUEST)
        assert (rec.vendor, rec.resources, rec.skipped, rec.cost_usd) == (
            GOOGLE_VERTEX,
            (),
            None,
            None,
        )
        assert {call[0] for call in auth.calls} == {GOOGLE_VERTEX}
    finally:
        await subject.aclose()


@pytest.mark.parametrize(
    "bad_grant",
    [
        None,
        OAuthAccess(access_token="fake-key", credential_id=1, kind="api_key"),
        OAuthAccess(access_token="fake-no-account", credential_id=1, kind="oauth"),
    ],
)
async def test_no_eligible_oauth_does_not_emit_inference(bare_manager, monkeypatch, bad_grant):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-subscription")
    monkeypatch.setenv("GOOGLE_API_KEY", "not-vertex-oauth")
    auth = Auth(openai=bad_grant, **{GOOGLE_VERTEX: None})
    posts = []

    def transport(http):
        posts.append(http.url.host)
        raise AssertionError("no eligible OAuth or native credential")

    subject = service(bare_manager, auth, transport)
    try:
        rec = await subject.recommend_resources(REQUEST)
        assert rec.skipped == "no-vendor"
        assert not posts
    finally:
        await subject.aclose()


async def test_real_auth_store_rejects_stored_provider_keys(bare_manager):
    store_row(bare_manager, "OPENAI_API_KEY", "fake-openai-key")
    store_row(bare_manager, "GOOGLE_VERTEX_API_KEY", "fake-vertex-key")
    posts = []

    def transport(http):
        posts.append(http.url.host)
        raise AssertionError("an API key must not reach the advisory inference call")

    subject = ClassificationService(config_dir=bare_manager)
    subject._http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    try:
        assert (await subject.recommend_resources(REQUEST)).skipped == "no-vendor"
        assert not posts
    finally:
        await subject.aclose()


@pytest.mark.parametrize("failure", ["401", "timeout", "unoffered", "unfinished", "refusal"])
async def test_unusable_coding_plan_hint_falls_through_to_native(bare_manager, failure):
    store_row(bare_manager, "TYPESAFE_API_KEY", "fake-native-key")
    posts = []
    in_flight = asyncio.Event()

    async def transport(http):
        posts.append(http.url.host)
        if http.url.host == "api.typesafe.ai":
            return native()
        assert http.url.host == "chatgpt.com"
        if failure == "401":
            return httpx.Response(401, text="do not log this")
        if failure == "timeout":
            with pytest.raises(asyncio.CancelledError):
                await in_flight.wait()
            raise asyncio.CancelledError
        if failure == "unoffered":
            return codex("uninstalled-resource")
        if failure == "unfinished":
            return codex(completed=False)
        return codex(part_type="refusal")

    subject = service(bare_manager, Auth(openai=OPENAI), transport, timeoutMs=400)
    try:
        rec = await subject.recommend_resources(REQUEST)
        assert rec.skipped is None and rec.vendor == "typesafe"
        assert rec.resources == tuple(ROSTER)
        assert posts == ["chatgpt.com", "api.typesafe.ai"]
    finally:
        await subject.aclose()


async def test_usable_radient_keeps_existing_native_roll_forward(bare_manager):
    store_row(bare_manager, "RADIENT_API_KEY", "fake-radient-key")
    store_row(bare_manager, "TYPESAFE_API_KEY", "fake-native-key")
    auth = Auth(openai=OPENAI, **{GOOGLE_VERTEX: VERTEX})
    posts = []

    def transport(http):
        posts.append(http.url.host)
        return httpx.Response(503) if http.url.host == "api.radienthq.com" else native()

    subject = service(bare_manager, auth, transport)
    try:
        rec = await subject.recommend_resources(REQUEST)
        assert rec.vendor == "typesafe"
        assert posts == ["api.radienthq.com", "api.typesafe.ai"]
        assert not auth.calls
    finally:
        await subject.aclose()


async def test_generated_route_is_never_reused_by_shared_typed_decide(bare_manager):
    auth = Auth(openai=OPENAI)
    posts = []

    def transport(http):
        posts.append(http.url.host)
        return codex()

    subject = service(bare_manager, auth, transport)
    try:
        assert (await subject.recommend_resources(REQUEST)).vendor == OPENAI_SUBSCRIPTION
        auth.calls.clear()
        for question in (
            Question("check", "noul", "", {}),
            Question("check", "score", "", ("a", "b")),
            Question("check", "choice", "", {"yes": "yes", "no": "no"}),
        ):
            assert await subject.decide(state="synthetic apple", question=question) is None
        assert not auth.calls
        assert posts == ["chatgpt.com"]
    finally:
        await subject.aclose()

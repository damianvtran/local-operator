"""The real recommendation cascade and adapters, with synthetic auth and HTTP."""

import asyncio
import dataclasses
import json
from types import SimpleNamespace

import httpx
import pytest

from local_operator.classification.advisory import GOOGLE_VERTEX, OPENAI_SUBSCRIPTION
from local_operator.classification.recommend import RecommendationRequest
from local_operator.classification.service import ClassificationService
from local_operator.classification.types import Answer, Question
from local_operator.classification.vendors import TypeSafeVendor, build_vendor
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.store import MeshAwareAuthStore
from local_operator.providers.auth_store import AuthStore, AuthStoreError, OAuthAccess
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

    async def get_oauth_access(
        self, provider, session_id, *, read_only, model_id, local_oauth_only
    ):
        assert local_oauth_only
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


def next_request(index):
    return dataclasses.replace(REQUEST, user_message=f"identify apple number {index}")


@pytest.mark.parametrize("source", ["env", "provider-key", "auth-key", "auth-oauth"])
async def test_local_radient_login_reclaims_a_warm_advisory_memo(bare_manager, monkeypatch, source):
    posts = []

    def transport(http):
        posts.append(http.url.host)
        if http.url.host == "chatgpt.com":
            return codex()
        assert http.headers["authorization"] == "Bearer new-local-credential"
        return native()

    auth = Auth(openai=OPENAI)
    subject = service(bare_manager, auth, transport)
    writer = AuthStore(bare_manager / "auth.db", config_dir=bare_manager)
    try:
        assert (await subject.recommend_resources(REQUEST)).vendor == OPENAI_SUBSCRIPTION
        if source == "env":
            monkeypatch.setenv("RADIENT_API_KEY", "new-local-credential")
        elif source == "provider-key":
            # SQLite/WAL-backed provider-class secret, not an auth.db row.
            store_row(bare_manager, "RADIENT_API_KEY", "new-local-credential")
        elif source == "auth-key":
            writer.upsert_credential(
                "radient", {"type": "api_key", "key": "new-local-credential", "source": "login"}
            )
        else:
            writer.upsert_credential("radient", {"type": "oauth", "access": "new-local-credential"})
        auth.calls.clear()
        assert (await subject.recommend_resources(next_request(1))).vendor == "radient"
        assert posts == ["chatgpt.com", "api.radienthq.com"]
        assert not auth.calls
    finally:
        await subject.aclose()
        writer.close()


async def test_login_during_awaited_oauth_lookup_prevents_stale_post(bare_manager, monkeypatch):
    resolving = asyncio.Event()
    resume = asyncio.Event()
    posts = []

    class PausedAuth(Auth):
        pause = False

        async def get_oauth_access(self, *args, **kwargs):
            if self.pause:
                # A distinct request with a warm no-Radient memo.
                resolving.set()
                await resume.wait()
            return await super().get_oauth_access(*args, **kwargs)

    def transport(http):
        posts.append(http.url.host)
        return codex() if http.url.host == "chatgpt.com" else native()

    auth = PausedAuth(openai=OPENAI)
    subject = service(bare_manager, auth, transport)
    try:
        assert (await subject.recommend_resources(REQUEST)).vendor == OPENAI_SUBSCRIPTION
        auth.pause = True
        async with asyncio.timeout(1):
            pending = asyncio.create_task(subject.recommend_resources(next_request(1)))
            await resolving.wait()
            monkeypatch.setenv("RADIENT_API_KEY", "new-during-resolve")
            resume.set()
            assert (await pending).vendor == "radient"
        assert posts == ["chatgpt.com", "api.radienthq.com"]
    finally:
        resume.set()
        await subject.aclose()


@pytest.mark.parametrize("warm", [False, True], ids=["cold-selection", "warm-oauth-miss"])
async def test_wal_radient_login_during_oauth_miss_wins_this_turn(bare_manager, warm):
    resolving = asyncio.Event()
    resume = asyncio.Event()
    posts = []
    db = bare_manager / "auth.db"
    reader = AuthStore(db, config_dir=bare_manager)
    writer = AuthStore(db, config_dir=bare_manager)
    row_id = writer.upsert_credential(
        "openai", {"type": "oauth", "access": "synthetic", "org_id": "synthetic"}
    ).id
    store_row(bare_manager, "TYPESAFE_API_KEY", "synthetic")

    class PausedAuth(MeshAwareAuthStore):
        pause = not warm

        async def get_oauth_access(self, provider, *args, **kwargs):
            if self.pause and provider == "openai":
                self.pause = False
                resolving.set()
                await resume.wait()
            return await super().get_oauth_access(provider, *args, **kwargs)

    def transport(http):
        posts.append(http.url.host)
        return codex() if http.url.host == "chatgpt.com" else native()

    auth = PausedAuth(reader, mesh=None, config_dir=bare_manager)
    subject = service(bare_manager, auth, transport)
    try:
        if warm:
            assert (await subject.recommend_resources(REQUEST)).vendor == OPENAI_SUBSCRIPTION
            auth.pause = True
        db_mtime = db.stat().st_mtime_ns
        async with asyncio.timeout(1):
            async with asyncio.TaskGroup() as tasks:
                pending = tasks.create_task(subject.recommend_resources(next_request(1)))
                await resolving.wait()
                writer.disable_credential(row_id, "fixture logout")
                writer.upsert_credential(
                    "radient",
                    {"type": "api_key", "key": "synthetic", "source": "login"},
                )
                assert db.stat().st_mtime_ns == db_mtime  # The change is in the WAL only.
                resume.set()
        assert pending.result().vendor == "radient", posts
        assert posts == ["chatgpt.com"] * warm + ["api.radienthq.com"]
    finally:
        await subject.aclose()
        auth.close()
        writer.close()


async def test_radient_login_during_native_401_reresolve_prevents_second_post(
    bare_manager, monkeypatch
):
    store_row(bare_manager, "TYPESAFE_API_KEY", "synthetic")
    resolving = asyncio.Event()
    resume = asyncio.Event()
    posts = []
    retrying = False
    original = TypeSafeVendor.credential

    async def credential(vendor, config_dir):
        if retrying:
            resolving.set()
            await resume.wait()
        return await original(vendor, config_dir)

    def transport(http):
        nonlocal retrying
        posts.append(http.url.host)
        if len(posts) == 2:
            retrying = True
            return httpx.Response(401)
        return native()

    monkeypatch.setattr(TypeSafeVendor, "credential", credential)
    subject = service(bare_manager, Auth(), transport)
    try:
        assert (await subject.recommend_resources(REQUEST)).vendor == "typesafe"
        async with asyncio.timeout(1):
            async with asyncio.TaskGroup() as tasks:
                pending = tasks.create_task(subject.recommend_resources(next_request(1)))
                await resolving.wait()
                monkeypatch.setenv("RADIENT_API_KEY", "synthetic")
                resume.set()
        assert pending.result().vendor == "radient", posts
        assert posts == ["api.typesafe.ai", "api.typesafe.ai", "api.radienthq.com"]
    finally:
        await subject.aclose()


@pytest.mark.parametrize("failure", ["401", "refusal", "timeout"])
async def test_advisory_breaker_does_not_open_native_typed_breaker(bare_manager, failure):
    posts = []
    blocked = asyncio.Event()

    async def transport(http):
        posts.append(http.url.host)
        if http.url.host == "api.typesafe.ai":
            return httpx.Response(
                200, json={"answers": {"check": {"choice": "yes", "confidence": 0.94}}}
            )
        assert http.url.host == "chatgpt.com"
        if failure == "timeout":
            await blocked.wait()
        if failure == "401":
            return httpx.Response(401)
        return codex(part_type="refusal")

    subject = service(bare_manager, Auth(openai=OPENAI), transport, timeoutMs=100)
    try:
        for index in range(3):
            rec = await subject.recommend_resources(next_request(index))
            assert rec.skipped == ("timeout" if failure == "timeout" else "error")
        assert posts.count("chatgpt.com") >= 3
        attempted = list(posts)
        assert (await subject.recommend_resources(next_request(3))).skipped == "no-vendor"
        assert posts == attempted  # Structured circuit actually stopped new HTTP.
        store_row(bare_manager, "TYPESAFE_API_KEY", "local-typesafe")
        question = Question("check", "choice", "", {"yes": "yes", "no": "no"})
        answer = await subject.decide(state="a fresh typed question", question=question)
        assert answer is not None and answer.value == "yes"
        assert posts == [*attempted, "api.typesafe.ai"]
    finally:
        await subject.aclose()


@pytest.mark.parametrize("source", ["borrow-only", "synced", "delegated", "owner-routing"])
async def test_advisory_never_borrows_or_spends_another_devices_grant(
    bare_manager, monkeypatch, source
):
    local = AuthStore(bare_manager / "auth.db", config_dir=bare_manager)
    if source == "owner-routing":
        local.upsert_credential(
            "openai", {"type": "oauth", "access": "local-but-not-selected", "org_id": "local"}
        )
    if source in {"synced", "delegated"}:
        origin = {"owner": "another-device", "key": "openai"}
        if source == "delegated":
            origin["grant_id"] = "delegation-id"
        local.upsert_credential(
            "openai",
            {"type": "oauth", "access": "remote", "org_id": "remote", "mesh_origin": origin},
        )
    auth = MeshAwareAuthStore(local, mesh=None, config_dir=bare_manager)

    if source == "owner-routing":
        auth.set_binding_reader(
            lambda _: SimpleNamespace(policy="owner", owner_device="another-device")
        )

    async def forbidden(*args, **kwargs):
        pytest.fail("automatic hints must not borrow or refresh another device's OAuth")

    monkeypatch.setattr(auth, "_borrow", forbidden)
    monkeypatch.setattr(local, "_ensure_oauth_fresh", forbidden)
    subject = service(bare_manager, auth, forbidden)
    try:
        assert (await subject.recommend_resources(REQUEST)).skipped == "no-vendor"
    finally:
        await subject.aclose()
        auth.close()


@pytest.mark.parametrize("expires", [True, False], ids=["moves-while-refresh-blocked", "fresh"])
async def test_advisory_uses_fresh_local_bearer_without_entering_refresh(
    bare_manager, monkeypatch, expires
):
    db = bare_manager / "auth.db"
    reader = AuthStore(db, config_dir=bare_manager)
    writer = AuthStore(db, config_dir=bare_manager)
    local = {
        "access": "synthetic-local",
        "refresh": "synthetic-local-refresh",
        "expires": 1 if expires else AuthStore._now_ms() + 3_600_000,
        # Delegation carries account_id, not org_id; use the same dedupe key.
        "account_id": "synthetic-org",
        "org_id": "synthetic-org",
    }
    row = writer.upsert_credential("openai", local)
    selected = asyncio.Event()
    refreshes = []
    posts = []

    class ObservedStore(MeshAwareAuthStore):
        async def get_oauth_access(self, provider, *args, **kwargs):
            if provider == "openai":
                selected.set()
            return await super().get_oauth_access(provider, *args, **kwargs)

    async def refresh(data):
        refreshes.append(data["refresh"])
        raise AuthStoreError("synthetic dispatch boundary")

    def transport(http):
        posts.append(http.url.host)
        return codex()

    # Use the production delegation writer; only its store supplier and the
    # signed-grant check are synthetic. The ordinary send fence would allow it.
    delegate_writer = MeshCredentialClient(root=bare_manager)
    monkeypatch.setattr(delegate_writer, "_auth_store", lambda: writer)
    reader.set_refresh_fence(lambda *_: None)
    monkeypatch.setattr(reader, "_refresh_fn", lambda _: refresh)
    auth = ObservedStore(reader, mesh=None, config_dir=bare_manager)
    subject = service(bare_manager, auth, transport)
    try:
        async with asyncio.timeout(1):
            async with asyncio.TaskGroup() as tasks:
                async with reader._refresh_lock_for(row.id):
                    pending = tasks.create_task(subject.recommend_resources(REQUEST))
                    await selected.wait()
                    if expires:
                        assert delegate_writer._write_delegated_row(
                            "openai",
                            owner="synthetic-owner",
                            provider="openai",
                            grant_id="synthetic-grant",
                            epoch=1,
                            data={
                                **local,
                                "access": "synthetic-delegated",
                                "refresh": "synthetic-delegated-refresh",
                            },
                        )
                        moved = writer.get_credential(row.id)
                        assert moved and moved.data["mesh_origin"]["grant_id"] == "synthetic-grant"
        assert refreshes == []
        if expires:
            assert pending.result().skipped == "no-vendor"
            assert posts == []
        else:
            assert pending.result().vendor == OPENAI_SUBSCRIPTION
            assert posts == ["chatgpt.com"]
    finally:
        await subject.aclose()
        auth.close()
        writer.close()


@pytest.mark.parametrize("project", [None, "persisted-vertex-project"])
async def test_vertex_only_bills_its_persisted_login_project(bare_manager, monkeypatch, project):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ambient-wrong-project")
    monkeypatch.setenv("VERTEX_PROJECT", "another-wrong-project")
    local = AuthStore(bare_manager / "auth.db", config_dir=bare_manager)
    local.upsert_credential(
        "google-vertex",
        {
            "type": "oauth",
            "access": "local-vertex",
            "project_id": project,
            "auth_mode": "oauth",
            "location": "us-central1",
        },
    )
    # Test the same adapter passed by a mesh-aware session, even for a local login.
    auth = MeshAwareAuthStore(local, mesh=None, config_dir=bare_manager)
    posts = []

    def transport(http):
        posts.append(http.url.host)
        assert f"/projects/{project}/" in http.url.path
        assert http.headers["x-goog-user-project"] == project
        assert http.headers["authorization"] == "Bearer local-vertex"
        return vertex()

    subject = service(bare_manager, auth, transport, preferred=GOOGLE_VERTEX)
    try:
        rec = await subject.recommend_resources(REQUEST)
        if project is None:
            assert rec.skipped == "no-vendor" and posts == []
        else:
            assert rec.vendor == GOOGLE_VERTEX and rec.resources == tuple(ROSTER)
            assert posts == ["us-central1-aiplatform.googleapis.com"]
    finally:
        await subject.aclose()
        auth.close()


async def test_advisory_success_does_not_reset_native_typed_failures(bare_manager):
    store_row(bare_manager, "TYPESAFE_API_KEY", "local-typesafe")
    posts = []

    def transport(http):
        posts.append(http.url.host)
        return codex() if http.url.host == "chatgpt.com" else httpx.Response(403)

    subject = service(bare_manager, Auth(openai=OPENAI), transport)
    question = Question("check", "choice", "", {"yes": "yes", "no": "no"})
    try:
        for index in range(3):
            if index:
                assert (await subject.recommend_resources(next_request(index))).vendor == (
                    OPENAI_SUBSCRIPTION
                )
            assert await subject.decide(state=f"typed {index}", question=question) is None
        assert posts.count("api.typesafe.ai") >= 3 and posts.count("chatgpt.com") == 2
        attempted = list(posts)
        assert await subject.decide(state="typed blocked", question=question) is None
        assert posts == attempted
    finally:
        await subject.aclose()

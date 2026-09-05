"""Route-local context defaults and ceilings must survive the runtime pipeline."""

import dataclasses

import pytest

from local_operator.compaction.thresholds import (
    CompactionSettings,
    resolve_threshold_tokens,
)
from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamModelEvent
from local_operator.model import configure, discovery
from local_operator.model.registry import ModelInfo
from local_operator.providers.auth_store import AuthStore, OAuthAccess
from local_operator.providers.failover import stream_with_failover
from local_operator.tui.widgets.model_picker import ModelPicker, ModelRow
from local_operator.tui.widgets.status_line import context_spelling
from tests.unit.providers.test_failover import FakeAuth, _FnClient


@pytest.fixture(autouse=True)
def isolated_metadata(monkeypatch):
    configure.invalidate_model_info_cache()
    monkeypatch.setattr(configure, "_from_price_catalogue", lambda p, m, info, **kw: info)
    monkeypatch.setattr(configure, "_from_aggregator_catalogue", lambda p, m, info, **kw: info)
    yield
    configure.invalidate_model_info_cache()


@pytest.mark.parametrize(
    "maximum,active",
    [(872000, 872000), (272000, 272000), (None, 272000), (-1, 272000), (True, 272000)],
)
def test_codex_default_max_round_trip(maximum, active):
    row = discovery._row_from_openai_codex_entry(
        {
            "slug": "gpt-6-astra",
            "context_window": 272000,
            "max_context_window": maximum,
        }
    )
    assert row is not None
    assert row.context_window == active
    assert row.default_context_window == 272000
    payload = {
        "capture": discovery.listing_capture_version("openai"),
        "models": [dataclasses.asdict(row)],
    }
    assert discovery._rows_from_payload(payload, discovery.listing_capture_version("openai")) == [
        row
    ]
    info = ModelInfo(id=row.id, name=row.id, description="public", context_window=1050000)
    merged = discovery._merge_one(row, info)
    assert merged.context_window == active
    assert merged.max_context_window == row.max_context_window
    assert merged.default_context_window == 272000


def catalogue(monkeypatch):
    calls = []

    def available(provider, **kwargs):
        calls.append((kwargs["is_oauth"], kwargs["account_id"]))
        maximum = 872000 if kwargs["account_id"] == "account-a" else 400000
        return [
            discovery.DiscoveredModel(
                id="gpt-5.6-sol",
                context_window=maximum,
                default_context_window=272000,
                max_context_window=maximum,
            )
        ], "ok"

    monkeypatch.setattr(discovery, "available_models", available)
    return calls


def test_complete_public_registry_does_not_bypass_account_catalogue(monkeypatch):
    calls = catalogue(monkeypatch)
    spec = ModelSpec(provider="openai", model_id="gpt-5.6-sol", context_window=1050000)
    a = OAuthAccess("secret-a", 1, account_id="account-a")
    b = OAuthAccess("secret-b", 2, account_id="account-b")
    first = configure.context_spec_for_access(spec, a, {})
    second = configure.context_spec_for_access(spec, b, {})
    refreshed = configure.context_spec_for_access(
        spec, dataclasses.replace(a, access_token="new-secret"), {}
    )
    assert (first.context_window, second.context_window, refreshed.context_window) == (
        872000,
        400000,
        872000,
    )
    assert calls == [(True, "account-a"), (True, "account-b")]
    default = configure.context_spec_for_access(
        spec, a, {"providers": {"openai": {"use_max_context_window": False}}}
    )
    assert default.context_window == 272000
    assert default.max_context_window == 872000
    public = configure.context_spec_for_access(first, OAuthAccess("api", 3, kind="api_key"), {})
    assert public.context_window == 1050000
    assert public.default_context_window is None
    assert public.max_context_window is None
    assert (
        resolve_threshold_tokens(
            first.context_window, CompactionSettings(threshold_percent=0.8, threshold_tokens=400000)
        )
        == 400000
    )
    assert context_spelling(300000, first.context_window) == "34.4%/872k"
    assert context_spelling(900000, first.context_window).startswith("103.2%")


def test_oauth_offline_never_adopts_public_ceiling(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "available_models",
        lambda *args, **kw: (
            [discovery.DiscoveredModel(id="gpt-5.6-sol", context_window=1050000)],
            "static",
        ),
    )
    spec = ModelSpec(provider="openai", model_id="gpt-5.6-sol", context_window=1050000)
    actual = configure.context_spec_for_access(
        spec, OAuthAccess("secret", 1, account_id="offline"), {}
    )
    assert actual.context_window == configure.UNKNOWN_CONTEXT_WINDOW
    assert actual.max_context_window is None


@pytest.mark.parametrize(
    "settings,expected",
    [
        (None, True),
        ({}, True),
        ({"providers": {"openai": {"use_max_context_window": False}}}, False),
        ({"providers": {"openai": {"use_max_context_window": "false"}}}, True),
    ],
)
def test_max_context_consumer_default(settings, expected):
    assert configure._openai_use_max_context_window(settings) is expected


def test_picker_distinguishes_provider_default_and_active_max():
    picker = ModelPicker(lambda row: None)
    row = ModelRow(
        provider="openai",
        model_id="gpt-6-astra",
        context_window=872000,
        default_context_window=272000,
        max_context_window=872000,
    )
    assert picker._window(row) == "872k max · provider default 272k"
    assert picker._window(row, compact=True) == "872k max"
    default = dataclasses.replace(row, context_window=272000)
    assert picker._window(default) == "provider default 272k active · 872k max"
    assert picker._window(default, compact=True) == "272k active"
    assert picker._window(dataclasses.replace(row, default_context_window=872000)) == "872k"


@pytest.mark.asyncio
@pytest.mark.parametrize("replayable", [False, True])
async def test_same_model_account_rotation_publishes_each_active_spec(monkeypatch, replayable):
    from local_operator.providers.failover import ProviderError

    catalogue(monkeypatch)

    class Accounts(FakeAuth):
        async def get_oauth_access(self, provider, session_id=None, **kwargs):
            key = await self.get_api_key(provider, session_id)
            return OAuthAccess(key, 1, account_id=key) if key else None

    auth = Accounts({"openai": ["account-a", "account-b"]})
    seen = []

    async def stream(request, key, access):
        seen.append((access.account_id, request.model.context_window))
        if access.account_id == "account-a":
            raise ProviderError(403, "quota exhausted")
        yield StreamEndEvent(stop_reason="stop")

    from local_operator.harness.types import ChatRequest

    request = ChatRequest(
        model=ModelSpec(provider="openai", model_id="gpt-5.6-sol"), replayable=replayable
    )
    events = [
        event
        async for event in stream_with_failover(
            request, auth, {"retry": {"baseDelayMs": 1}}, lambda spec: _FnClient(stream)
        )
    ]
    assert seen == [("account-a", 872000), ("account-b", 400000)]
    assert [
        event.model.context_window for event in events if isinstance(event, StreamModelEvent)
    ] == [872000, 400000]


@pytest.mark.asyncio
async def test_session_adopts_request_metadata_without_changing_compaction(tmp_path):
    from local_operator.harness.types import ModelChangeEvent, StreamTextDelta
    from tests.unit.session.test_active_route import RoutedStream, _session

    before = ModelSpec(provider="openai", model_id="gpt-6-astra", context_window=272000)
    active = before.model_copy(
        update={
            "context_window": 872000,
            "default_context_window": 272000,
            "max_context_window": 872000,
        }
    )
    stream = RoutedStream(
        [
            [
                StreamModelEvent(model=active),
                StreamTextDelta(delta="ready"),
                StreamEndEvent(stop_reason="stop"),
            ]
        ]
    )
    session = _session(tmp_path, stream, model=before, has_ui=True)
    events = []
    session.subscribe(events.append)
    await session.prompt("Describe the active context.")
    assert session.effective_model.context_window == 872000
    assert session.effective_model.default_context_window == 272000
    assert [event.context_window for event in events if isinstance(event, ModelChangeEvent)] == [
        872000
    ]
    projected = session.frontend_state.effective_model
    assert projected is not None
    assert projected.context_window == 872000
    await session.dispose()


@pytest.mark.asyncio
async def test_cold_resume_does_not_restore_pre_maximum_window(tmp_path, monkeypatch):
    from local_operator.config import ConfigManager
    from local_operator.providers import failover
    from local_operator.session.frontend_state import (
        FrontendModelSpec,
        FrontendSessionState,
    )
    from local_operator.session.remote import RemoteSession

    catalogue(monkeypatch)
    config = ConfigManager(config_dir=tmp_path)
    config.set_config_value("hosting", "openai")
    config.set_config_value("model_name", "gpt-5.6-sol")

    async def access(*args, **kwargs):
        assert kwargs["read_only"] is True
        return OAuthAccess("secret-a", 1, account_id="account-a")

    monkeypatch.setattr(failover, "_resolve_access_for_provider", access)
    remote = RemoteSession(config_dir=tmp_path, session_id="cold", takeover_factory=lambda: None)
    state = await remote._synthesise_cold_state(str(tmp_path))
    assert state.selected_model is not None
    assert state.selected_model.context_window == 872000
    durable = FrontendSessionState(
        session_id="cold",
        epoch="old",
        selected_model=FrontendModelSpec(
            provider="openai", model_id="gpt-5.6-sol", context_window=272000
        ),
    )
    assert remote._restored_model_specs(state, durable) == {}
    config.set_config_value("providers", {"openai": {"use_max_context_window": False}})
    state = await remote._synthesise_cold_state(str(tmp_path))
    assert state.selected_model is not None
    assert state.selected_model.context_window == 272000
    assert state.selected_model.max_context_window == 872000


def test_catalogue_current_entry_uses_serving_spec(tmp_path):
    from local_operator.providers.controller import ProviderController

    auth = AuthStore(tmp_path / "catalogue-auth.db")
    controller = ProviderController(auth)
    spec = ModelSpec(
        provider="openai",
        model_id="gpt-5.6-sol",
        context_window=872000,
        default_context_window=272000,
        max_context_window=872000,
    )
    entry = controller.entry_for("openai", "gpt-5.6-sol", spec=spec)
    assert entry is not None
    auth.close()
    assert entry.context_window == 872000
    assert entry.default_context_window == 272000
    assert entry.max_context_window == 872000


def test_missing_account_never_reads_public_api_cache(tmp_path, monkeypatch):
    from tests.unit.model.test_discovery import _Response, _StubClient

    discovery.available_models(
        "openai",
        api_key="api-key",
        cache_dir=tmp_path,
        client=_StubClient(
            [_Response(200, {"data": [{"id": "gpt-5.6-sol", "context_window": 1050000}]})]
        ),
    )
    rows, status = discovery.available_models(
        "openai", api_key="oauth", is_oauth=True, cache_dir=tmp_path, client=_StubClient([])
    )
    assert status == "static"
    assert all(row.context_window == 0 for row in rows)


def test_api_route_recovers_public_limit_after_unavailable_oauth(monkeypatch):
    monkeypatch.setattr(discovery, "available_models", lambda *args, **kwargs: ([], "static"))
    public = ModelSpec(provider="openai", model_id="gpt-5.6-sol", context_window=1050000)
    unknown = configure.context_spec_for_access(
        public, OAuthAccess("oauth", 1, account_id="missing"), {}
    )
    assert unknown.context_window == 128000
    assert unknown.default_context_window is None
    assert unknown.max_context_window is None
    restored = configure.context_spec_for_access(unknown, OAuthAccess("api", 2, kind="api_key"), {})
    assert restored.context_window == 1050000
    assert restored.context_metadata_resolved


@pytest.mark.asyncio
@pytest.mark.parametrize("access_kind", ["offline", "missing-account", "missing-auth"])
async def test_fresh_unknown_cold_state_rejects_legacy_capacity(tmp_path, monkeypatch, access_kind):
    from local_operator.config import ConfigManager
    from local_operator.providers import failover
    from local_operator.session.frontend_state import (
        FrontendModelSpec,
        FrontendSessionState,
    )
    from local_operator.session.remote import RemoteSession

    monkeypatch.setattr(discovery, "available_models", lambda *args, **kwargs: ([], "static"))
    config = ConfigManager(config_dir=tmp_path)
    config.set_config_value("hosting", "openai")
    config.set_config_value("model_name", "gpt-5.6-sol")

    async def resolve(*args, **kwargs):
        if access_kind == "missing-auth":
            return None
        return OAuthAccess("oauth", 1, account_id="account" if access_kind == "offline" else None)

    monkeypatch.setattr(failover, "_resolve_access_for_provider", resolve)
    remote = RemoteSession(config_dir=tmp_path, session_id="cold", takeover_factory=lambda: None)
    state = await remote._synthesise_cold_state(str(tmp_path))
    assert state.selected_model is not None
    assert state.selected_model.context_window == 128000
    assert state.selected_model.context_metadata_resolved
    legacy = FrontendSessionState(
        session_id="cold",
        epoch="legacy",
        selected_model=FrontendModelSpec(
            provider="openai", model_id="gpt-5.6-sol", context_window=1050000
        ),
    )
    assert remote._restored_model_specs(state, legacy) == {}
    # JSON snapshots must preserve provenance through attach/replay, including
    # the absence of positive provider limit metadata.
    restored = FrontendSessionState.model_validate_json(state.model_dump_json())
    assert remote._restored_model_specs(restored, legacy) == {}


@pytest.mark.asyncio
async def test_cold_auth_lifetime_stays_on_one_worker(tmp_path, monkeypatch):
    import asyncio
    import threading

    from local_operator.config import ConfigManager
    from local_operator.providers import auth_store, failover
    from local_operator.session.remote import RemoteSession

    owners = []

    class ThreadCheckedAuth(AuthStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.owner = threading.get_ident()
            owners.append(self.owner)

        def close(self):
            assert threading.get_ident() == self.owner
            super().close()

    async def resolve(auth, *args, **kwargs):
        assert threading.get_ident() == auth.owner
        # This real SQLite read catches both event-loop and executor handoffs.
        auth.list_credentials("openai")
        await asyncio.sleep(0)
        return None

    monkeypatch.setattr(auth_store, "AuthStore", ThreadCheckedAuth)
    monkeypatch.setattr(failover, "_resolve_access_for_provider", resolve)
    config = ConfigManager(config_dir=tmp_path)
    config.set_config_value("hosting", "openai")
    config.set_config_value("model_name", "gpt-5.6-sol")
    barrier = threading.Barrier(4)
    await asyncio.gather(*(asyncio.to_thread(barrier.wait) for _ in range(4)))
    remote = RemoteSession(config_dir=tmp_path, session_id="cold", takeover_factory=lambda: None)
    state = await remote._synthesise_cold_state(str(tmp_path))
    assert state.selected_model is not None
    assert state.selected_model.context_metadata_resolved
    assert owners and owners[0] != threading.get_ident()


def test_wire_identity_wins_over_other_account_field(monkeypatch):
    catalogue(monkeypatch)
    spec = ModelSpec(provider="openai", model_id="gpt-5.6-sol")
    access = OAuthAccess("token", 1, account_id="account-b", org_id="account-a")
    assert configure.context_spec_for_access(spec, access, {}).context_window == 872000


@pytest.mark.asyncio
async def test_primary_recovery_keeps_new_context_metadata(tmp_path):
    from local_operator.harness.types import ModelChangeEvent
    from tests.unit.session.test_active_route import RoutedStream, _session

    primary = ModelSpec(provider="openai", model_id="gpt-6-astra", context_window=272000)
    session = _session(tmp_path, RoutedStream(), model=primary)
    session._active_fallback = ModelSpec(
        provider="anthropic", model_id="other", context_window=200000
    )
    await session._emit(
        ModelChangeEvent(
            provider="openai",
            model_id="gpt-6-astra",
            context_window=872000,
            default_context_window=272000,
            max_context_window=872000,
            context_metadata=True,
        )
    )
    await session._on_route_settled(None, "recovered")
    assert session.effective_model.context_window == 872000
    await session.dispose()


def test_child_stats_use_active_context_without_resolving_again():
    from local_operator.harness.types import Usage
    from local_operator.tui.widgets.subagent_panel import job_stats
    from tests.unit.tui.test_subagent_stats import Job

    child = Job(
        model_label="openai/gpt-6-astra", context_window=872000, usage=Usage(input_tokens=300000)
    )
    stats = job_stats(child)
    assert context_spelling(stats.context_tokens, stats.context_window) == "34.4%/872k"

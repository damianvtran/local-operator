"""Account/root/native metadata regressions: synthetic data, never live inference."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.model import configure, discovery
from local_operator.model.registry import ModelInfo
from local_operator.session.model_admission import (
    RuntimeUpgradeNeededError,
    require_owner_provider,
)


def test_meta_selected_credential_is_scoped_in_existing_resolver(monkeypatch, tmp_path):
    observed = []

    def resolve(provider, ident, bucket, scope, base):
        observed.append((configure._listing_credential.get(), scope, base))
        return ModelInfo(id=ident, name=ident, description="fixture", context_window=1007997)

    monkeypatch.setattr(configure, "_resolve_model_info_cached", resolve)
    a = ("LLM|1|synthetic_a", True, "account-a")
    b = ("LLM|2|synthetic_b", True, "account-b")
    for credential in (a, b):
        info = configure.resolve_model_info(
            "meta-muse", "muse-fixture", credential=credential, base=tmp_path
        )
        assert info.context_window == 1007997
    assert [entry[0] for entry in observed] == [a, b]
    assert observed[0][1] != observed[1][1]
    assert all(entry[2] == tmp_path for entry in observed)
    assert configure._listing_credential.get() is None


def test_meta_selected_spec_uses_native_context_and_effort_not_codex_budget(monkeypatch, tmp_path):
    seen = []

    def info(provider, ident, *, credential, base):
        seen.append((credential, base))
        return ModelInfo(
            id=ident,
            name="Native fixture",
            description="fixture",
            context_window=1007997,
            max_tokens=128000,
            reasoning=True,
            supports_tools=False,
            reasoning_default_effort="high",
        )

    monkeypatch.setattr(configure, "resolve_model_info", info)
    # The real resolver warms the listing memo while reading the account
    # document (``info_from_discovered_model`` -> ``_remember_listing_effort``);
    # this stub replaces that read, so the warm is explicit. The ladder must
    # NOT ride ModelInfo: a non-scalar field breaks the paint path's
    # shallow-copy isolation (tests/unit/model/test_catalogue.py pins it).
    configure._store_listing_effort("meta-muse", "muse-fixture", ("medium", "high"))
    spec = configure.ModelSpec(
        provider="meta-muse", model_id="muse-fixture", context_window=128000, reasoning_effort="max"
    )
    access = SimpleNamespace(
        access_token="LLM|1|synthetic",
        kind="oauth",
        org_id=None,
        account_id=None,
        email="fixture@example.test",
    )
    result = configure.context_spec_for_access(
        spec, access, {"openai_use_max_context_window": False}, base=tmp_path
    )
    assert result.context_window == 1007997 and result.max_output_tokens == 128000
    assert result.reasoning_efforts == ["medium", "high"] and result.reasoning_effort == "high"
    assert result.supports_tools is False
    assert seen == [((access.access_token, True, access.email), tmp_path)]


def test_native_meta_projection_and_cache_roundtrip_preserve_details():
    entry = {
        "id": "muse-fixture",
        "metadata": {
            "muse-code": {
                "name": "Provider native name",
                "family": "native-family",
                "release_date": "2026-10-12",
                "is_hidden": False,
                "reasoning": True,
                "tool_call": False,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 1007997, "output": 128000},
                "variants": {"medium": {}, "high": {"description": "fixture high"}},
            }
        },
    }
    import dataclasses

    row = discovery._row_from_meta_muse_entry(entry)
    assert row is not None and row.name == "Provider native name"
    decoded = discovery._rows_from_payload(
        {
            "capture": discovery.listing_capture_version("meta-muse"),
            "models": [dataclasses.asdict(row)],
        },
        discovery.listing_capture_version("meta-muse"),
    )
    assert decoded == [row]
    assert row.family == "native-family" and row.max_tokens == 128000
    assert row.variants == {"medium": {}, "high": {"description": "fixture high"}}


@pytest.mark.parametrize("native,stale,positive", [(None, True, ()), ([], False, ())])
def test_new_builtin_on_unaffirmed_owner_refuses_without_signin_advice(native, stale, positive):
    with pytest.raises(RuntimeUpgradeNeededError) as caught:
        require_owner_provider(
            "meta-muse",
            established={"openai"},
            native=native,
            positive_catalogue=positive,
            stale=stale,
        )
    assert caught.value.code == "runtime_upgrade_needed"
    assert "saved sign-in is unchanged" in str(caught.value)


@pytest.mark.parametrize(
    "established,native,positive,provider",
    [
        ({"meta-muse"}, None, (), "meta-muse"),
        ({"openai"}, ["meta-muse"], (), "meta-muse"),
        ({"openai"}, None, ("meta-muse",), "meta-muse"),
        ({"openai"}, None, (), "fixture-custom-compatible-host"),
    ],
)
def test_known_provider_evidence_or_custom_host_preserves_existing_admission(
    established, native, positive, provider
):
    require_owner_provider(
        provider, established=established, native=native, positive_catalogue=positive, stale=True
    )


def test_meta_listing_lookup_binds_explicit_database_and_sealed_root(monkeypatch, tmp_path):
    from local_operator.providers import auth_store

    observed = []
    row = SimpleNamespace(
        data={"account_id": "fixture-account", "access_secret_ref": "opaque-fixture"},
        credential_type="oauth",
        disabled_cause=None,
    )

    class FakeStore:
        def list_credentials(self, provider):
            assert provider == "meta-muse"
            return [row]

        def _oauth_key_fn(self, provider):
            assert provider == "meta-muse"
            return lambda data: "LLM|1|synthetic"

    def shared(*, db_path, config_dir):
        observed.append((db_path, config_dir))
        return FakeStore()

    monkeypatch.setattr(auth_store, "shared_auth_store", shared)
    result = configure._oauth_listing_token("meta-muse", base=tmp_path)
    assert result == ("LLM|1|synthetic", True, "fixture-account")
    assert observed == [(tmp_path / "auth.db", tmp_path)]


def test_saved_but_unusable_access_is_not_claimed_signed_out():
    from local_operator.session.frontend_state import model_access_claim

    claim = model_access_claim("meta-muse/muse-wonder-1.0", set(), saved={"meta-muse"})
    assert claim is not None and claim.state == "temporarily_unavailable"
    missing = model_access_claim("meta-muse/muse-wonder-1.0", set())
    assert missing is not None and missing.state == "signed_out"


@pytest.mark.asyncio
async def test_session_factory_dispatch_store_binds_database_and_secrets_to_explicit_root(
    tmp_path, monkeypatch
):
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.network import credentials
    from local_operator.session_factory import create_session
    from tests.unit.test_session_factory import _args

    ambient, explicit = tmp_path / "ambient", tmp_path / "explicit"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(ambient))
    original = credentials.build_auth_store
    captured = []

    def build(root, *, db_path):
        store = original(root, db_path=db_path)
        captured.append((root, db_path, store._db_path, store._config_dir))
        return store

    monkeypatch.setattr(credentials, "build_auth_store", build)
    session = await create_session(
        _args(hosting="test", model="test", yolo=True),
        ConfigManager(explicit),
        AgentRegistry(explicit),
    )
    try:
        # Other boot integrations may still consult their legacy ambient store;
        # this regression pins the SESSION's dispatch store, not global defaults.
        assert captured == [(explicit, explicit / "auth.db", explicit / "auth.db", explicit)]
        assert (explicit / "auth.db").exists()
    finally:
        await session.dispose()


def test_picker_detail_is_selected_native_plain_text_with_bounded_cell_width():
    from local_operator.model.ranking import ModelRow
    from local_operator.tui.widgets.model_picker import ModelPicker

    widget = ModelPicker(on_choose=lambda row: None)
    widget.set_rows(
        [
            ModelRow(
                provider="meta-muse",
                model_id="muse-fixture",
                label="Native fixture",
                max_output_tokens=128000,
                supports_tools=False,
                reasoning=True,
                reasoning_efforts=("medium", "high"),
                family="native family",
                release_date="2026-10-12",
                modalities={"input": ["text", "image"]},
            )
        ]
    )
    widget.open("muse-fixture")
    rows = widget._native_detail_rows(100)
    assert len(rows) == 3
    text = "\n".join(row.plain for row in rows)
    assert "output 128,000" in text and "tools no" in text and "medium/high" in text
    assert "released 2026-10-12" in text
    assert all(row.cell_len <= 35 for row in widget._native_detail_rows(35))


def test_the_canonical_six_rung_detail_line_fits_the_pickers_widest_cap():
    """D1: the six-rung line clipped mid-value at the 73-cell widest cap.

    The finder's canonical shape (six effort variants, text/image in, text out)
    measured 74 cells against a 73-cell content width at 100 columns — full at
    80, ``… · output te…`` at 100 — so the modality labels give six cells back
    and the whole line fits the cap it is widest against.
    """
    from local_operator.model.ranking import ModelRow
    from local_operator.tui.widgets.model_picker import ModelPicker

    widget = ModelPicker(on_choose=lambda row: None)
    widget.set_rows(
        [
            ModelRow(
                provider="meta-muse",
                model_id="muse-fixture",
                label="Native fixture",
                max_output_tokens=128000,
                supports_tools=True,
                supports_images=True,
                reasoning=True,
                reasoning_efforts=("minimal", "low", "medium", "high", "xhigh", "max"),
                family="fixture-family",
                release_date="2026-10-12",
                modalities={"input": ["text", "image"], "output": ["text"]},
            )
        ]
    )
    widget.open("muse-fixture")
    rows = widget._native_detail_rows(73)
    assert len(rows) == 3
    detail = rows[1].plain
    assert "in text/image" in detail and "out text" in detail
    assert "\u2026" not in detail, detail
    assert rows[1].cell_len <= 73


@pytest.mark.asyncio
async def test_attached_owner_refusal_preserves_primary_and_new_bind_is_rechecked():
    import asyncio

    from local_operator.session.attached import AttachedSession

    primary = configure.ModelSpec(provider="openai", model_id="fixture-old", context_window=128000)
    requested = configure.ModelSpec(
        provider="meta-muse", model_id="muse-fixture", context_window=1007997
    )
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.session.runtime.types import SessionRecord

    calls = []

    class Client(AttachClient):
        def __init__(self):
            # This seam records owner RPC admission, never opens a socket.
            pass

        async def set_model(self, provider: str, model_id: str, effort: str | None = None) -> str:
            calls.append((provider, model_id))
            return "accepted"

    class Facade(AttachedSession):
        @property
        def model(self):
            return primary.model_copy()

        @property
        def effective_model(self):
            return primary.model_copy()

        def _read_state_field(self, name):
            assert name == "model_catalogue"
            return []  # Truncated/absent inventory is not a negative support claim.

    facade = object.__new__(Facade)
    facade._client = Client()
    facade._runtime_record = SessionRecord(
        pid=12345,
        kind="daemon",
        session_id="synthetic-meta-owner",
        conversation_name="fixture",
        cwd="synthetic-workspace",
        model_label="openai/fixture-old",
        control_port=0,
        control_key="synthetic-unused-control",
        version="old-fixture",
    )
    with pytest.raises(RuntimeUpgradeNeededError) as caught:
        facade.set_model(requested, explicit=True)
    assert caught.value.support == "unverified"
    assert calls == [] and facade.model.provider == "openai"
    # A newly bound owner's positive claim is consumed, not cached at refusal.
    facade._runtime_record = SessionRecord(
        pid=12346,
        kind="daemon",
        session_id="synthetic-meta-owner",
        conversation_name="fixture",
        cwd="synthetic-workspace",
        model_label="openai/fixture-old",
        control_port=0,
        control_key="synthetic-unused-control",
        version="new-fixture",
        native_provider_ids=["meta-muse"],
    )
    facade.set_model(requested, explicit=True)
    await asyncio.sleep(0)
    assert calls == [("meta-muse", "muse-fixture")]
    assert facade.model.provider == "openai"  # Only the actual owner can mutate it.

"""Authenticated Muse native metadata, visibility and account-cache isolation."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from local_operator.model.configure import build_model_spec
from local_operator.model.discovery import available_models
from local_operator.model.registry import static_models
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import PICKER_TTL_S, ProviderController
from local_operator.providers.meta_muse_credentials import delete_key

KEY = "LLM|1234|fixture_subscription_key"


def row(
    model_id,
    *,
    hidden=False,
    output="text",
    variants=("minimal", "low", "medium", "high", "xhigh", "max"),
):
    return {
        "id": model_id,
        "metadata": {
            "muse-code": {
                "is_hidden": hidden,
                "tool_call": True,
                "reasoning": True,
                "modalities": {"input": ["text", "image"], "output": [output]},
                "limit": {"context": 1007997, "output": 128000},
                "variants": {effort: {} for effort in variants},
            }
        },
    }


def test_subscription_catalogue_projects_actual_nested_fields_and_discovered_full(tmp_path: Path):
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == "https://api.meta.ai/muse-code/models"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert request.headers["x-api-version"] == "1.0.0"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    row("muse-wonder-1.0"),
                    row("muse-wonder-full-1.0"),
                    row("muse-wonder-1.0-contributor"),
                    row("muse-image-1.0", hidden=True, output="image"),
                    row("muse-hidden-1.0", hidden=True),
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        rows, status = available_models(
            "meta-muse", api_key=KEY, is_oauth=True, cache_dir=tmp_path, client=http
        )
    assert status == "ok"
    assert {model.id for model in rows} == {"muse-wonder-1.0", "muse-wonder-full-1.0"}
    full = next(model for model in rows if "full" in model.id)
    assert full.context_window == 1007997 and full.max_tokens == 128000
    assert full.supports_images and full.supports_tools
    assert full.reasoning_efforts == ("minimal", "low", "medium", "high", "xhigh", "max")
    assert len(calls) == 1
    assert "muse-wonder-full-1.0" not in static_models("meta-muse")
    assert "chatgpt" not in str(calls[0].url)
    assert KEY not in "".join(path.name for path in tmp_path.iterdir())


def test_a_shortened_ttl_refetches_a_young_muse_listing_and_a_default_one_reuses_it(
    tmp_path: Path,
):
    """Fresh-on-open is the picker's SHORTENED TTL, decided in the reader.

    ``docs/META_MUSE.md`` ("Catalogue freshness and existing conversations")
    promises that a user-opened picker refreshes Muse's authenticated listing even
    when its successful cache is young. A caller that shortens the hard TTL is
    asking for an answer fresher than any reuse window can promise, which for Muse
    means a real GET; the catalogue controller passes ``PICKER_TTL_S`` through
    untouched (``test_the_picker_ttl_is_the_only_ttl_override`` pins that), so the
    rule cannot live in the controller as a TTL override.

    The control is the DEFAULT read: an unshortened TTL must keep reusing the
    document, or every picker keystroke would block on the network.
    """
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"data": [row("muse-wonder-1.0")]})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        _rows, warm = available_models(
            "meta-muse", api_key=KEY, is_oauth=True, cache_dir=tmp_path, client=http
        )
        assert (warm, len(calls)) == ("ok", 1)

        _rows, bumped = available_models(
            "meta-muse",
            api_key=KEY,
            is_oauth=True,
            cache_dir=tmp_path,
            client=http,
            ttl_s=PICKER_TTL_S,
        )
        assert bumped == "ok"
        assert len(calls) == 2, "a picker-open read must not be served from a young cache"

        available_models("meta-muse", api_key=KEY, is_oauth=True, cache_dir=tmp_path, client=http)
        assert len(calls) == 2, "an unshortened TTL must keep reusing the document"


def test_empty_eligible_catalogue_is_authoritative_not_static_ghosts(tmp_path: Path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, json={"data": [row("muse-image-1.0", hidden=True, output="image")]}
            )
        )
    ) as http:
        rows, _ = available_models(
            "meta-muse",
            api_key=KEY,
            is_oauth=True,
            account_id="scope",
            cache_dir=tmp_path,
            client=http,
        )
    assert rows == []


@pytest.mark.asyncio
async def test_email_only_signin_drives_listing_and_missing_secret_is_unusable(tmp_path: Path):
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        saved = store.upsert_credential(
            "meta-muse",
            {
                "type": "oauth",
                "access": KEY,
                "email": "fixture@example.test",
                "source": "device-approval",
            },
        )
        controller = ProviderController(store, tmp_path)
        secret, is_oauth, scope = await controller._listing_credential("meta-muse")
        assert secret == KEY and is_oauth and scope == "fixture@example.test"
        assert controller.has_any_credential("meta-muse")
        delete_key(saved.data, tmp_path)
        assert not controller.has_any_credential("meta-muse")
        usable = controller.usable_providers()
        assert usable is not None
        assert "meta-muse" not in usable
    finally:
        store.close()


def test_wonder_spec_uses_responses_and_verified_effort_not_full_alias():
    spec = build_model_spec("meta-muse", "muse-wonder-1.0")
    assert spec.supports_responses_api
    assert spec.reasoning_efforts == ("minimal", "low", "medium", "high", "xhigh", "max")
    assert "none" not in spec.reasoning_efforts

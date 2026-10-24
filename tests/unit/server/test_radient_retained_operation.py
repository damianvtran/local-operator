"""Trusted retained_notice is additive; operation.message stays sanitized."""

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import ProviderController
from local_operator.providers.oauth.callback_server import LoginError
from local_operator.providers.oauth.radient import RadientLoginDisabled
from local_operator.server.utils import desktop_auth
from tests.unit.providers import test_radient_account_retention as _retention

store = _retention.store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["success", "arbitrary_error", "disabled", "empty", "other_provider"]
)
async def test_operation_publishes_only_trusted_retained_notice(
    store: AuthStore, monkeypatch, outcome
):
    old = store.upsert_credential(
        "radient", {"type": "oauth", "access": "synthetic", "refresh": "synthetic"}
    )
    store.disable_credential(
        old.id, "operator_disabled" if outcome == "empty" else "identity_unattributed"
    )
    effects = []

    async def login(self, provider_id, **kwargs):
        if outcome == "arbitrary_error":
            raise LoginError("UNTRUSTED-provider-secret")
        if outcome == "disabled":
            raise RadientLoginDisabled()
        return "UNTRUSTED-controller-return-and-grant-note"

    monkeypatch.setattr(ProviderController, "login", login)
    monkeypatch.setattr(
        desktop_auth,
        "apply_desktop_login_defaults",
        lambda *a, **kw: effects.append("defaults") or {},
    )
    host = desktop_auth.DesktopAuth(store, store.db_path.parent)

    async def rehome(provider):
        effects.append("rehome")
        return [], []

    host.rehome = rehome
    try:
        operation = await host.start("anthropic" if outcome == "other_provider" else "radient")
        assert operation.task is not None
        await operation.task
        published = operation.snapshot()
        assert "UNTRUSTED" not in published["message"]
        assert "UNTRUSTED" not in (published.get("retained_notice") or "")
        if outcome in {"success", "empty", "other_provider"}:
            assert published["state"] == "succeeded"
            assert published["message"] == "Sign-in complete."
            if outcome == "success":
                notice = published.get("retained_notice")
                assert isinstance(notice, str)
                assert "Account could not be verified" in notice
                assert f"[{old.id}]" in notice
                assert "local-operator login radient" in notice
            else:
                assert published.get("retained_notice") is None
            assert effects == ["defaults", "rehome"]
        else:
            assert published.get("retained_notice") is None
            assert published["state"] == "failed"
            if outcome == "disabled":
                assert "remains unavailable" in published["message"]
                assert "saved disabled state was preserved" in published["message"]
            else:
                assert published["message"] == "Sign-in failed. Check the provider and try again."
            assert effects == []
    finally:
        await host.close()

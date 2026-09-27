"""The TUI's recovery hint wiring: one helper, two disjoint remedies.

``OperatorApp._with_recovery_hint`` is the ONE helper every surface of the app
routes a turn error through, so these tests pin what it returns for the three
cases the Radient workstream added or could disturb: a Radient quota error
gains the usage-limit remedy; a Radient AUTH error keeps its existing
``/login`` hint and does not gain the quota one (no double-fire); and any other
provider's quota error stays byte-identical to the rendered error.

The store and the probe seam are injected — no test reads the operator's
credentials or the network — and the app object is built the same way
``test_usage_continuity`` builds it, without a running Textual app.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_operator.providers import radient_recovery as rr
from local_operator.providers.auth_store import AuthStore
from local_operator.tui.app import OperatorApp

RENDERED_QUOTA = "rate limit or quota exceeded (HTTP 402): insufficient credits"
RENDERED_AUTH = "authentication failed (HTTP 401): bad key"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """No test may read the operator's store or environment for a key."""
    import local_operator.providers.registry as registry

    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    monkeypatch.setattr(registry, "provider_secret_value", lambda *args, **kwargs: None)
    rr.reset_recovery_cache()
    yield
    rr.reset_recovery_cache()


def _app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, arm_probe: bool, calls: list[str]):
    import local_operator.providers.auth_store as auth_store_module

    store = AuthStore(tmp_path / "auth.db")
    if arm_probe:
        store.upsert_credential("radient", {"type": "oauth", "access": "tok-1", "refresh": "r"})

        def probe(token: str):
            calls.append(token)
            return rr.parse_verification({"signup_grant": "pending", "grant_amount": 5})

        monkeypatch.setattr(rr, "_probe_verification_sync", probe)
    monkeypatch.setattr(auth_store_module, "shared_auth_store", lambda *args, **kwargs: store)

    app = OperatorApp(lambda: None)  # type: ignore[arg-type]
    app._session = MagicMock(model_label="radient/auto")
    return app


def test_a_radient_quota_error_gains_the_usage_limit_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)

    out = app._with_recovery_hint(RENDERED_QUOTA)

    assert out.startswith(RENDERED_QUOTA)
    assert "check your email" in out and "$5.00" in out
    assert "/login" not in out, "the quota remedy must not borrow the auth remedy"
    assert calls == ["tok-1"]


def test_a_radient_auth_error_keeps_its_login_hint_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disjointness the display sites depend on: one kind, one remedy."""
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)

    out = app._with_recovery_hint(RENDERED_AUTH)

    assert "/login radient" in out
    assert "check your email" not in out
    assert calls == [], "an auth error must not spend a Radient usage probe"


def test_another_providers_quota_error_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)
    app._session = MagicMock(model_label="openai/gpt-5")

    assert app._with_recovery_hint(RENDERED_QUOTA) == RENDERED_QUOTA
    assert calls == []

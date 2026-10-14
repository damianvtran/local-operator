"""H5 remediation — the SHARED offline authority fence (S-1/H5-4) and the
ORIGINAL-scope validator (S-3).

Matrix rules exercised here (security preflight rows 1/2/4-part/9-reads):
a disabled/revoked/drifted/consentless runtime refuses with a TYPED sentence
and performs no fetch; reads stay distinguished from mutations; an
empty-generation scope is non-authorizing. These are pure-fake unit tests: the
Store and the config reader are substituted at their module attributes (the
fence imports both at CALL time, which is why monkeypatching module attrs
works without touching the implementation).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge import store as store_mod
from local_operator.team_bridge.authority import (
    authority_fence,
    scope_from_runtime,
    scope_matches_runtime,
)

pytestmark = pytest.mark.asyncio


def _owner(**over: object) -> SimpleNamespace:
    base = dict(
        canonical_base_url="https://hub.example",
        account_id="acc_1",
        tenant_id="ten_1",
        endpoint_id="ep_1",
        credential_id=7,
        enrollment_generation=3,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _runtime(owner: object | None = None, **over: object) -> SimpleNamespace:
    base = dict(
        owner=owner if owner is not None else _owner(),
        config_dir="/tmp/fence-test",
        tenant_id="ten_1",
        session_id="sess-1",
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Store:
    """Scripted store: the fence's four reads are the whole surface."""

    def __init__(self, *, owner: object | None, live: bool = True, consent: object = "c1"):
        self._owner = owner
        self._live = live
        self._consent = consent

    def active_owner(self) -> object | None:
        return self._owner

    def current_generation_is_live(self, owner: object) -> bool:
        return self._live

    def owner_consent_receipt(self, owner: object) -> object:
        return self._consent

    def designated_session_for(self, owner: object) -> str:
        return "sess-1"


@pytest.fixture()
def _enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "local_operator.config.read_config_values",
        lambda config_dir: {"team_bridge": {"enabled": True}},
    )
    monkeypatch.setattr(store_mod, "Store", lambda config_dir: _Store(owner=_owner()))


async def test_feature_off_refuses_with_zero_network(
    _enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "local_operator.config.read_config_values",
        lambda config_dir: {"team_bridge": {"enabled": False}},
    )
    refusal = authority_fence(_runtime())
    assert refusal is not None and refusal["code"] == "feature_off"


async def test_torn_config_read_is_not_still_enabled(
    _enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # read_config_values returns None for a file it cannot parse RIGHT NOW:
    # the fence must fail closed rather than treat a torn read as "unset".
    monkeypatch.setattr(
        "local_operator.config.read_config_values", lambda config_dir: None
    )
    refusal = authority_fence(_runtime())
    assert refusal is not None and refusal["code"] == "feature_off"


async def test_missing_or_ambiguous_owner_is_owner_changed(
    _enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_mod, "Store", lambda config_dir: _Store(owner=None))
    refusal = authority_fence(_runtime())
    assert refusal is not None and refusal["code"] == "owner_changed"


async def test_empty_generation_is_non_authorizing(
    _enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _owner(enrollment_generation=0)
    monkeypatch.setattr(store_mod, "Store", lambda config_dir: _Store(owner=owner))
    refusal = authority_fence(_runtime(owner=owner))
    assert refusal is not None and refusal["code"] == "owner_changed"


async def test_dead_generation_is_owner_changed(
    _enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        store_mod, "Store", lambda config_dir: _Store(owner=_owner(), live=False)
    )
    refusal = authority_fence(_runtime())
    assert refusal is not None and refusal["code"] == "owner_changed"


async def test_consent_loss_is_typed_consent_revoked(
    _enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        store_mod, "Store", lambda config_dir: _Store(owner=_owner(), consent=None)
    )
    refusal = authority_fence(_runtime())
    assert refusal is not None and refusal["code"] == "consent_revoked"


@pytest.mark.parametrize(
    "drift",
    [
        {"canonical_base_url": "https://other.example"},
        {"account_id": "acc_2"},
        {"tenant_id": "ten_2"},
        {"endpoint_id": "ep_2"},
        {"credential_id": 8},
        {"enrollment_generation": 4},
    ],
)
async def test_binding_drift_is_owner_changed(
    _enabled: None, monkeypatch: pytest.MonkeyPatch, drift: dict
) -> None:
    live = _owner(**drift)
    # The committed side is the DRIFTED one; the runtime still holds the old
    # binding — either direction must refuse (no silent rebind).
    monkeypatch.setattr(store_mod, "Store", lambda config_dir: _Store(owner=live))
    refusal = authority_fence(_runtime())
    assert refusal is not None and refusal["code"] == "owner_changed"


async def test_tenant_inconsistency_and_designation_change_refuse(
    _enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert authority_fence(_runtime(tenant_id="ten_OTHER")) is not None
    monkeypatch.setattr(
        store_mod,
        "Store",
        lambda config_dir: type(
            "_S",
            (),
            {
                "active_owner": lambda self, owner=_owner(): owner,
                "current_generation_is_live": lambda self, owner: True,
                "owner_consent_receipt": lambda self, owner: "c1",
                "designated_session_for": lambda self, owner: "sess-OTHER",
            },
        )(),
    )
    refusal = authority_fence(_runtime())
    assert refusal is not None and refusal["code"] == "owner_changed"


async def test_everything_holds_passes(_enabled: None) -> None:
    assert authority_fence(_runtime()) is None


async def test_scope_from_runtime_round_trips_and_validates() -> None:
    runtime = _runtime()
    scope = scope_from_runtime(runtime)
    assert scope_matches_runtime(scope, runtime) is True
    # EMPTY generation is non-authorizing — never backfilled from current.
    assert scope_matches_runtime({**scope, "generation": ""}, runtime) is False
    assert scope_matches_runtime({**scope, "credential_id": 99}, runtime) is False
    assert scope_matches_runtime({**scope, "session_id": "sess-2"}, runtime) is False
    assert scope_matches_runtime(None, runtime) is False

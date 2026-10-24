"""Peer-create credential preflight, through the real relay and auth store.

The requester cannot see the peer's inventory, so these cells configure the OWNING
relay's root and read the reply it composes. Synthetic registry ids keep the test
independent of every real provider account; the registry, config and SQLite store
under test are the production implementations, not doubles of the preflight seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.network import identity
from local_operator.providers.auth_store import AuthStore
from tests.unit.network.test_model_choice_frontdoor import _frame, _link, _warmed_server
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — peer_pair's fixture
    devices,
)
from tests.unit.network.test_session_plane import (
    _dial_to,
    _listen,
    _pair,
    _serve,
    _stop_all,
)

HELD = "mesh-held-synthetic"
MISSING = "mesh-missing-synthetic"
MODEL = "synthetic-chat-model"


@pytest.fixture(autouse=True)
def _synthetic_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register chat providers with no environment credential or live network route."""
    from local_operator.providers import registry

    definitions = [
        registry.ProviderDefinition(id=provider, name=provider) for provider in (HELD, MISSING)
    ]
    for definition in definitions:
        # _BY_ID is the registry's compiled lookup, exactly what the production
        # alias resolver reads; monkeypatch restores it after each cell.
        monkeypatch.setitem(registry._BY_ID, definition.id, definition)  # noqa: SLF001
    # ProviderController imported the list object itself, so patch ITS binding too;
    # adding only _BY_ID would make the store row real but invisible to its census.
    from local_operator.providers import controller

    monkeypatch.setattr(
        controller,
        "PROVIDER_REGISTRY",
        [*controller.PROVIDER_REGISTRY, *definitions],
    )


@pytest.fixture()
def _live_peer_pair(request: pytest.FixtureRequest) -> Any:
    """Resolve the imported ``devices`` fixture without shadowing its module binding."""
    return request.getfixturevalue("devices")


def _configure_default(root: Path, provider: str) -> None:
    ConfigManager(config_dir=root).update_config({"hosting": provider, "model_name": MODEL})


def _hold_credential(
    root: Path, provider: str = HELD, payload: dict[str, Any] | None = None
) -> None:
    """Write one synthetic row through the real store; never read its value back."""
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        auth.upsert_credential(
            provider,
            payload or {"type": "api_key", "source": "login", "key": "synthetic-fixture-only"},
        )
    finally:
        auth.close()


def _create(root: Path, **fields: Any) -> dict[str, Any]:
    identity.mint(root, name="cloud-node-1")
    return _warmed_server(root)._op_session_create(_link(), _frame(**fields))  # noqa: SLF001


def _assert_refused_before_mint(root: Path, reply: dict[str, Any], sentence: str) -> None:
    assert reply["op"] == "error"
    assert reply["code"] == "model_credential_missing"
    assert reply["message"] == sentence
    assert reply["detail"]["model"] == {"applied": False, "detail": sentence}
    assert not (root / "sessions").exists(), "a refused create must leave no session or claim"


def test_an_unserviceable_default_refuses_create_with_the_peers_sentence(root: Path) -> None:
    _configure_default(root, MISSING)

    reply = _create(root, prompt="run on the default")

    sentence = (
        f"the resolved default model uses provider '{MISSING}', but this device holds no "
        "credential for it. Create with --model <provider>/<model-id> naming a provider "
        "this device holds, or share that credential to this device first with "
        f"'lop network credential share {MISSING} --with <peer>'."
    )
    _assert_refused_before_mint(root, reply, sentence)


def test_an_explicit_unheld_provider_is_refused_even_when_the_default_is_held(
    root: Path,
) -> None:
    _configure_default(root, HELD)
    _hold_credential(root)

    reply = _create(root, model={"provider": MISSING, "model_id": MODEL})

    sentence = (
        f"the requested model uses provider '{MISSING}', but this device holds no credential "
        "for it. Create with --model <provider>/<model-id> naming a provider this device "
        "holds, or share that credential to this device first with "
        f"'lop network credential share {MISSING} --with <peer>'."
    )
    _assert_refused_before_mint(root, reply, sentence)


def test_an_explicit_held_provider_escapes_an_unserviceable_default(root: Path) -> None:
    """A fresh peer's bad default must not make every peer-create unusable."""
    _configure_default(root, MISSING)
    _hold_credential(root)

    reply = _create(root, prompt="", model={"provider": HELD, "model_id": MODEL})

    assert reply["session_id"]
    assert reply["warming"] is True
    assert reply["model"] == {
        "applied": False,
        "detail": "the runtime is joining; the model is applied when it arrives",
    }


def test_a_login_flavour_uses_its_canonical_held_credential(root: Path) -> None:
    """xai-oauth must see the real xai OAuth row, not refuse its raw spelling."""
    _configure_default(root, MISSING)
    _hold_credential(
        root,
        "xai",
        {"access": "synthetic-access-only", "refresh": "synthetic-refresh-only"},
    )

    reply = _create(root, prompt="", model={"provider": "xai-oauth", "model_id": MODEL})

    assert reply["session_id"]
    assert reply["warming"] is True
    assert reply["model"]["detail"] == (
        "the runtime is joining; the model is applied when it arrives"
    )


@pytest.mark.parametrize("provider", ["radient", "radient-key"])
def test_the_radient_family_gets_a_model_only_remedy(root: Path, provider: str) -> None:
    _configure_default(root, provider)

    reply = _create(root, prompt="run on the default")

    sentence = (
        f"the resolved default model uses provider '{provider}', but this device holds no "
        "credential for it. Create with --model <provider>/<model-id> naming a provider "
        "this device holds instead."
    )
    _assert_refused_before_mint(root, reply, sentence)
    assert "credential share" not in sentence
    assert "declined" not in sentence


def test_the_refusal_and_model_sentence_cross_a_real_peer_link(
    _live_peer_pair: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wire preserves both the failed-create envelope and its structured sentence."""
    server_a, server_b, _host_a, _port_a = _live_peer_pair
    _configure_default(server_b.root, MISSING)
    record, _host, _port = _pair(_live_peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    served = _serve(monkeypatch, server_b.root)
    try:
        link = _dial_to(server_a, record, host_b, port_b)
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 51,
                "locality": "remote",
                "cwd": str(server_b.root),
                "prompt": "this must not be admitted",
            }
        )
        sentence = (
            f"the resolved default model uses provider '{MISSING}', but this device holds no "
            "credential for it. Create with --model <provider>/<model-id> naming a provider "
            "this device holds, or share that credential to this device first with "
            f"'lop network credential share {MISSING} --with <peer>'."
        )
        assert reply is not None and reply["op"] == "error", reply
        assert reply["code"] == "model_credential_missing"
        assert reply["message"] == sentence
        assert reply["detail"]["model"] == {"applied": False, "detail": sentence}
        assert not (server_b.root / "sessions").exists()
        link.close("test")
    finally:
        _stop_all(served)

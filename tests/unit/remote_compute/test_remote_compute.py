"""The remote-compute preview flag and the network binding every run needs.

Isolation: every test writes its OWN config root and its OWN mesh store under
``tmp_path``. The flag reader's no-argument form is exercised with
``LOCAL_OPERATOR_CONFIG_DIR`` pointed at that temp root, so a run can never read
the operator's live ``~/.local-operator`` (AGENTS.md, "Isolating a run").
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from local_operator import remote_compute
from local_operator.config import ConfigManager
from local_operator.network import store
from local_operator.network.types import MemberRecord, MeshRefusal, NetworkRecord
from local_operator.remote_compute import (
    CODE_AMBIGUOUS_NETWORK,
    CODE_DUPLICATE_NETWORK_ID,
    CODE_INVALID_NETWORK_ID,
    CODE_NO_NETWORK,
    CODE_NOT_A_MEMBER,
    CODE_UNKNOWN_NETWORK,
    NetworkBinding,
    RunCreateRequest,
    resolve_network,
)
from local_operator.server.features import feature_flags


def _network(
    name: str, network_id: str | None = None, *, self_active: bool = True
) -> NetworkRecord:
    """One local network record with this device's own member row."""
    record = NetworkRecord(
        network_id=network_id or store.new_network_id(),
        name=name,
        self_device_id="d_self",
        members=[
            MemberRecord(
                device_id="d_self",
                name="this device",
                removed_at=None if self_active else 1.0,
                removed_by=None if self_active else "d_admin",
            )
        ],
    )
    return record


@pytest.fixture()
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private config dir, exported so ``is_enabled()``'s default path lands here."""
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


def _write_flag(config_root: Path, enabled: bool) -> None:
    ConfigManager(config_root).set_config_value("remote_compute", {"enabled": enabled})


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------


def test_the_flag_defaults_off(config_root: Path) -> None:
    assert remote_compute.is_enabled() is False
    assert remote_compute.is_enabled(config_root) is False
    # Absent must mean OFF even in a config file that exists and has other keys.
    ConfigManager(config_root).set_config_value("model", "gpt-5")
    assert remote_compute.is_enabled(config_root) is False


def test_a_non_boolean_value_does_not_turn_the_flag_on(config_root: Path) -> None:
    # Only a real YAML boolean enables it: this switch decides whether work
    # leaves the device, so a hand-edited "yes" or 1 is not coerced.
    ConfigManager(config_root).set_config_value("remote_compute", {"enabled": "yes"})
    assert remote_compute.is_enabled(config_root) is False
    ConfigManager(config_root).set_config_value("remote_compute", {"enabled": 1})
    assert remote_compute.is_enabled(config_root) is False


def test_an_unreadable_config_falls_back_to_off(config_root: Path) -> None:
    (config_root / "config.yml").write_text("values: [unclosed\n", encoding="utf-8")
    assert remote_compute.is_enabled(config_root) is False


def test_the_capability_bit_follows_a_live_toggle(config_root: Path) -> None:
    assert "remote_compute" not in feature_flags()
    _write_flag(config_root, True)
    # No reload, no new manager: feature_flags() re-reads config.yml per call.
    assert feature_flags()["remote_compute"] == 1
    _write_flag(config_root, False)
    assert "remote_compute" not in feature_flags()


def test_the_enabling_key_is_the_registry_path() -> None:
    from local_operator import settings_io

    setting = settings_io.BY_KEY["remote_compute.enabled"]
    assert setting.path == remote_compute.SETTING_PATH
    assert setting.default is remote_compute.DEFAULT_REMOTE_COMPUTE_ENABLED
    assert setting.section == "remote_compute"


# ---------------------------------------------------------------------------
# Network resolution
# ---------------------------------------------------------------------------


@pytest.fixture()
def mesh_root(tmp_path: Path) -> Path:
    return tmp_path / "mesh"


def _save(record: NetworkRecord, root: Path) -> None:
    store.save(record, root)


def test_resolve_by_id(mesh_root: Path) -> None:
    record = _network("home")
    _save(record, mesh_root)
    binding = resolve_network(record.network_id, mesh_root)
    assert binding == NetworkBinding(network_id=record.network_id, name="home")


def test_resolve_by_name(mesh_root: Path) -> None:
    record = _network("home")
    _save(record, mesh_root)
    assert resolve_network("home", mesh_root).network_id == record.network_id


def test_resolve_id_wins_over_a_same_named_other_network(mesh_root: Path) -> None:
    # Names are not unique by design; an exact id must never be read as a name.
    first = _network("home")
    second = _network("home")
    _save(first, mesh_root)
    _save(second, mesh_root)
    assert resolve_network(first.network_id, mesh_root).network_id == first.network_id


def test_unknown_network_refuses(mesh_root: Path) -> None:
    _save(_network("home"), mesh_root)
    with pytest.raises(MeshRefusal) as caught:
        resolve_network("office", mesh_root)
    assert caught.value.code == CODE_UNKNOWN_NETWORK


def test_an_id_this_device_never_joined_is_unknown_not_a_guess(mesh_root: Path) -> None:
    _save(_network("home"), mesh_root)
    with pytest.raises(MeshRefusal) as caught:
        resolve_network(store.new_network_id(), mesh_root)
    assert caught.value.code == CODE_UNKNOWN_NETWORK


def test_ambiguous_name_refuses_and_names_the_remedy(mesh_root: Path) -> None:
    _save(_network("home"), mesh_root)
    _save(_network("home"), mesh_root)
    with pytest.raises(MeshRefusal) as caught:
        resolve_network("home", mesh_root)
    assert caught.value.code == CODE_AMBIGUOUS_NETWORK
    assert "use the network id" in caught.value.sentence


def test_no_target_and_no_network_refuses(mesh_root: Path) -> None:
    with pytest.raises(MeshRefusal) as caught:
        resolve_network(None, mesh_root)
    assert caught.value.code == CODE_NO_NETWORK
    assert "lop network init" in caught.value.sentence


def test_no_target_and_one_network_is_that_network(mesh_root: Path) -> None:
    record = _network("home")
    _save(record, mesh_root)
    assert resolve_network(None, mesh_root).network_id == record.network_id


def test_no_target_and_two_networks_refuses_rather_than_picking_a_tenant(
    mesh_root: Path,
) -> None:
    first = _network("home")
    second = _network("work")
    _save(first, mesh_root)
    _save(second, mesh_root)
    with pytest.raises(MeshRefusal) as caught:
        resolve_network(None, mesh_root)
    assert caught.value.code == CODE_AMBIGUOUS_NETWORK
    # Both candidates are named, so the reader can choose without `lop network ls`.
    assert first.network_id in caught.value.sentence
    assert second.network_id in caught.value.sentence


def test_a_removed_membership_refuses(mesh_root: Path) -> None:
    record = _network("home", self_active=False)
    _save(record, mesh_root)
    with pytest.raises(MeshRefusal) as caught:
        resolve_network("home", mesh_root)
    assert caught.value.code == CODE_NOT_A_MEMBER


def test_an_id_that_is_not_the_minted_shape_refuses(mesh_root: Path) -> None:
    # Older records keep their id verbatim, but this field is what the server
    # keys a tenant by, so a non-minted id must not be sent.
    record = _network("legacy", network_id="legacy-network")
    _save(record, mesh_root)
    with pytest.raises(MeshRefusal) as caught:
        resolve_network("legacy", mesh_root)
    assert caught.value.code == CODE_INVALID_NETWORK_ID


# ---------------------------------------------------------------------------
# The request model
# ---------------------------------------------------------------------------


def test_a_request_carries_the_binding_network(mesh_root: Path) -> None:
    record = _network("home")
    _save(record, mesh_root)
    binding = resolve_network("home", mesh_root)
    request = RunCreateRequest.for_binding(binding, {"command": ["echo", "hi"]})
    assert request.to_json() == {
        "command": ["echo", "hi"],
        "network_id": record.network_id,
    }


def test_a_request_cannot_be_built_without_a_network_id() -> None:
    with pytest.raises(TypeError):
        RunCreateRequest()  # type: ignore[call-arg]


def test_a_request_refuses_an_unminted_network_id() -> None:
    with pytest.raises(MeshRefusal) as caught:
        RunCreateRequest(network_id="home")
    assert caught.value.code == CODE_INVALID_NETWORK_ID


def test_a_body_may_not_restate_the_tenant() -> None:
    binding = NetworkBinding(network_id=store.new_network_id(), name="home")
    with pytest.raises(MeshRefusal) as caught:
        RunCreateRequest.for_binding(binding, {"network_id": store.new_network_id()})
    assert caught.value.code == CODE_DUPLICATE_NETWORK_ID


def test_the_binding_is_frozen() -> None:
    binding = NetworkBinding(network_id=store.new_network_id(), name="home")
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.network_id = store.new_network_id()  # type: ignore[misc]

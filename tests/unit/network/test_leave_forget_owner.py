"""Member departure forgets only that network's observations about that owner."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator import supervisors
from local_operator.network import relay, store, types
from local_operator.network.credentials.state import PlacementState

SELF = "d_" + "1" * 32
OWNER = "d_" + "2" * 32
OTHER = "d_" + "3" * 32
NETWORK = "n_" + "1" * 32
SECOND_NETWORK = "n_" + "2" * 32


@pytest.fixture()
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    attempts: list[str] = []
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / "config"))
    monkeypatch.chdir(home)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        attempts.append("host control")
        raise AssertionError("departure test attempted host control")

    class ForbiddenPopen(subprocess.Popen[bytes]):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            forbidden(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", ForbiddenPopen)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden)
    monkeypatch.setattr(supervisors, "systemctl_user", forbidden)
    monkeypatch.setattr(supervisors, "schtasks", forbidden)
    monkeypatch.setattr(relay, "_launchctl", forbidden)
    real_write = store._write_private_json

    def guarded_write(path: Path, payload: Any) -> Path:
        if not path.is_relative_to(tmp_path):
            attempts.append("outside fixture write")
            raise AssertionError("departure test attempted outside fixture write")
        return real_write(path, payload)

    monkeypatch.setattr(store, "_write_private_json", guarded_write)
    outside = tmp_path.parent / "synthetic-denied-write.json"
    with pytest.raises(AssertionError, match="outside fixture write"):
        store._write_private_json(outside, {"schema": 1})
    with pytest.raises(AssertionError, match="host control"):
        os.kill(424242, 0)
    with pytest.raises(AssertionError, match="host control"):
        supervisors.systemctl_user("status", "synthetic-never-live")
    assert attempts == ["outside fixture write", "host control", "host control"]
    assert not outside.exists()
    attempts.clear()
    yield home / "config"
    assert attempts == [], "a swallowed forbidden write/control attempt still fails"
    print("negative controls refused; outside writes/host-control attempts after controls: []")


def _record(root: Path) -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="synthetic-net",
        self_device_id=SELF,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    record.members = [types.MemberRecord(device_id=device) for device in (SELF, OWNER, OTHER)]
    store.save(record, root)
    return record


def _seed(root: Path, network: str) -> PlacementState:
    state = PlacementState(network, root=root)
    for key, owner in (("openai", OWNER), ("anthropic", OWNER), ("gemini", OTHER)):
        state.observe(key, "owner_offline", owner_device=owner, retry_after_ms=3_600_000)
    until = time.time() + 3600
    for owner, row_id in ((OWNER, 7), (OWNER, 8), (OTHER, 7), (OTHER, 8)):
        state.cool_down(
            "openai",
            owner_device=owner,
            credential_id=row_id,
            block_scope="account",
            until=until,
        )
    state.save()
    return state


@pytest.mark.parametrize("persist", [True, False])
def test_public_leave_forgets_only_departing_owner_in_its_network(
    isolated_root: Path, persist: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(isolated_root)
    state = _seed(isolated_root, NETWORK)
    other_network = _seed(isolated_root, SECOND_NETWORK)
    other_bytes = other_network.path.read_bytes()
    before = state.to_json()
    decoy = _seed(isolated_root.parent / "decoy-config", NETWORK)
    decoy_bytes = decoy.path.read_bytes()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(decoy.root))

    relay.leave(record, device_id=OWNER, root=isolated_root, persist=persist)

    after = PlacementState.load(NETWORK, isolated_root).to_json()
    assert decoy.path.read_bytes() == decoy_bytes, "use the caller's root, not the environment"
    if persist:
        saved = store.load(NETWORK, isolated_root)
        assert saved is not None
        departed, untouched = saved.member(OWNER), saved.member(OTHER)
        assert departed is not None and untouched is not None
        assert not departed.active and untouched.active
        print("readback: membership departure persisted; checking owner-scoped cache cleanup")
        assert all(row["owner_device"] != OWNER for row in after["observations"])
        assert all(row["owner_device"] != OWNER for row in after.get("cooldowns", []))
        assert after["observations"] == [
            r for r in before["observations"] if r["owner_device"] == OTHER
        ]
        assert after["cooldowns"] == [r for r in before["cooldowns"] if r["owner_device"] == OTHER]
        assert {row["credential_id"] for row in after["cooldowns"]} == {7, 8}
    else:
        assert after == before, "non-persisting member-table calculations must not write cache"
    assert other_network.path.read_bytes() == other_bytes
    print(
        f"public leave persist={persist}: observations="
        f"{len(before['observations'])}->{len(after['observations'])}; cooldowns="
        f"{len(before.get('cooldowns', []))}->{len(after.get('cooldowns', []))}; "
        "other owner/accounts/network/root preserved"
    )


def test_unknown_member_does_not_forget_any_owner(isolated_root: Path) -> None:
    record = _record(isolated_root)
    state = _seed(isolated_root, NETWORK)
    before = state.path.read_bytes()
    with pytest.raises(types.MeshRefusal, match="not a member"):
        relay.leave(record, device_id="d_" + "9" * 32, root=isolated_root)
    assert state.path.read_bytes() == before


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_cache_failure_cannot_interrupt_committed_departure(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    record = _record(isolated_root)
    state = _seed(isolated_root, NETWORK)
    before = state.path.read_bytes()

    def fail_save(self: PlacementState) -> Path:
        raise failure("synthetic cache failure")

    monkeypatch.setattr(PlacementState, "save", fail_save)
    departed = relay.leave(record, device_id=OWNER, root=isolated_root)
    assert not departed.active
    saved = store.load(NETWORK, isolated_root)
    assert saved is not None
    member = saved.member(OWNER)
    assert member is not None and not member.active
    assert state.path.read_bytes() == before
    print(f"public leave: {failure.__name__} in cache save does not undo membership departure")


@pytest.mark.parametrize("unrelated_cache", [False, True])
def test_departure_does_not_create_or_rewrite_an_unrelated_cache(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch, unrelated_cache: bool
) -> None:
    record = _record(isolated_root)
    state = PlacementState(NETWORK, root=isolated_root)
    if unrelated_cache:
        state.observe("gemini", "owner_offline", owner_device=OTHER, retry_after_ms=3_600_000)
        state.save()
    before = state.path.read_bytes() if state.path.exists() else None

    def forbidden_save(self: PlacementState) -> Path:
        raise AssertionError("no target-owner rows: do not write a cache")

    monkeypatch.setattr(PlacementState, "save", forbidden_save)
    relay.leave(record, device_id=OWNER, root=isolated_root)
    assert (state.path.read_bytes() if state.path.exists() else None) == before


def test_failed_departure_does_not_forget_cached_observations(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(isolated_root)
    state = _seed(isolated_root, NETWORK)
    before = state.path.read_bytes()
    calls: list[str] = []

    def fail_departure(*args: Any, **kwargs: Any) -> None:
        calls.append("save")
        raise OSError("synthetic membership persistence failure")

    def forbidden_forget(self: PlacementState, device: str) -> None:
        calls.append("forget")
        raise AssertionError("failed departure must not reach owner-cache cleanup")

    monkeypatch.setattr(store, "save", fail_departure)
    monkeypatch.setattr(PlacementState, "forget_owner", forbidden_forget)
    with pytest.raises(OSError, match="membership persistence failure"):
        relay.leave(record, device_id=OWNER, root=isolated_root)
    assert calls == ["save"]
    assert state.path.read_bytes() == before
    saved = store.load(NETWORK, isolated_root)
    assert saved is not None
    member = saved.member(OWNER)
    assert member is not None and member.active
    print("failed departure: member still active on disk; cache byte-identical; forget calls=0")

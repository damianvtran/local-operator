"""Member departure forgets only that network's observations about that owner."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import supervisors
from local_operator.network import relay, store, types
from local_operator.network.credentials import state as state_module
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
    real_stage = store._stage_private

    def guarded_stage(path: Path, emit: Any) -> Path:
        if not path.is_relative_to(tmp_path):
            attempts.append("outside fixture write")
            raise AssertionError("departure test attempted outside fixture write")
        return real_stage(path, emit)

    monkeypatch.setattr(store, "_stage_private", guarded_stage)
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
    record.members = [
        types.MemberRecord(device_id=device, role="admin" if device == SELF else "drive")
        for device in (SELF, OWNER, OTHER)
    ]
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
    helper_calls: list[str] = []
    if not persist:

        def forbidden_helper(*args: Any, **kwargs: Any) -> None:
            helper_calls.append("evict")
            raise AssertionError("non-persisting leave must not invoke cache eviction")

        monkeypatch.setattr(
            PlacementState, "forget_persisted_owner", forbidden_helper, raising=False
        )

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
        assert helper_calls == []
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

    real_write = store._stage_private

    def fail_cache_write(path: Path, payload: Any) -> Path:
        if path == state.path:
            fail_save(state)
        return real_write(path, payload)

    monkeypatch.setattr(store, "_stage_private", fail_cache_write)
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

    cache_writes: list[Path] = []
    real_write = store._stage_private

    def guard_cache_write(path: Path, payload: Any) -> Path:
        if path == state.path:
            cache_writes.append(path)
            forbidden_save(state)
        return real_write(path, payload)

    monkeypatch.setattr(store, "_stage_private", guard_cache_write)
    relay.leave(record, device_id=OWNER, root=isolated_root)
    assert cache_writes == []
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

    def forbidden_forget(*args: Any, **kwargs: Any) -> None:
        calls.append("forget")
        raise AssertionError("failed departure must not reach owner-cache cleanup")

    monkeypatch.setattr(store, "save", fail_departure)
    monkeypatch.setattr(PlacementState, "forget_persisted_owner", forbidden_forget, raising=False)
    with pytest.raises(OSError, match="membership persistence failure"):
        relay.leave(record, device_id=OWNER, root=isolated_root)
    assert calls == ["save"]
    assert state.path.read_bytes() == before
    saved = store.load(NETWORK, isolated_root)
    assert saved is not None
    member = saved.member(OWNER)
    assert member is not None and member.active
    print("failed departure: member still active on disk; cache byte-identical; forget calls=0")


@pytest.mark.parametrize(
    "failure", ["none", "symlink", "OSError", "ValueError", "RuntimeError", "KeyboardInterrupt"]
)
def test_real_leave_handler_rotates_despite_cache_failure(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    _record(isolated_root)
    state = _seed(isolated_root, NETWORK)
    before = state.path.read_bytes()
    target = isolated_root.parent / "CACHE_PATH_CANARY"
    refusals: list[str] = []
    real_guard = state_module._ensure_private_dir

    def record_guard(*args: Any, **kwargs: Any) -> None:
        try:
            real_guard(*args, **kwargs)
        except types.MeshRefusal as exc:
            refusals.append(exc.code)
            raise

    monkeypatch.setattr(state_module, "_ensure_private_dir", record_guard)
    if failure == "symlink":
        state.path.parent.rename(target)
        state.path.parent.symlink_to(target, target_is_directory=True)
    elif failure != "none":
        real_write = store._stage_private
        kind = {
            "OSError": OSError,
            "ValueError": ValueError,
            "RuntimeError": RuntimeError,
            "KeyboardInterrupt": KeyboardInterrupt,
        }[failure]

        def fail_write(path: Path, payload: Any) -> Path:
            if path == state.path:
                raise kind(f"CACHE_PAYLOAD_CANARY {isolated_root}")
            return real_write(path, payload)

        monkeypatch.setattr(store, "_stage_private", fail_write)
    store.save_secrets(
        types.SecretState(network_id=NETWORK, epoch=1, secret=relay.wire.b64u(os.urandom(32))),
        isolated_root,
    )
    audits: list[str] = []
    broadcasts: list[str] = []
    server: Any = SimpleNamespace(
        root=isolated_root,
        store_view=SimpleNamespace(network=lambda _id: store.load(NETWORK, isolated_root)),
        audit=SimpleNamespace(record=lambda event: audits.append(event.event)),
        _broadcast_epoch=lambda _record, _state, **kw: broadcasts.append(kw["reason"]),
    )
    caplog.set_level(logging.WARNING, logger=relay.logger.name)
    link: Any = SimpleNamespace(network_id=NETWORK, device_id=OWNER)
    if failure == "KeyboardInterrupt":
        with pytest.raises(KeyboardInterrupt):
            relay.RelayServer._op_leave(server, link, {"req": "synthetic"})
        assert not caplog.records, "control signals must propagate, not be logged/swallowed"
        return
    reply = relay.RelayServer._op_leave(server, link, {"req": "synthetic"})
    assert reply["detail"] == {"left": OWNER}
    saved = store.load(NETWORK, isolated_root)
    assert saved is not None and saved.epoch == 2
    assert store.require_secrets(NETWORK, isolated_root).epoch == 2
    assert audits == broadcasts == ["member_left"]
    if failure == "symlink":
        assert refusals == ["credential_path_invalid"], "the real namespace guard must still refuse"
    warnings = [
        r for r in caplog.records if r.name == relay.logger.name and r.levelno >= logging.WARNING
    ]
    if failure == "none":
        assert warnings == []
    else:
        assert state.path.read_bytes() == before
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert NETWORK in message and len(message) < 256
        assert ("MeshRefusal" if failure == "symlink" else failure) in message
        assert "CANARY" not in message and str(isolated_root) not in message
        assert not warnings[0].exc_info
    print(f"real _op_leave {failure}: epoch 1->2; audit=1; broadcast=1; warnings={len(warnings)}")


def test_owner_eviction_preserves_update_committed_under_shared_cache_lock(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(isolated_root)
    state = _seed(isolated_root, NETWORK)
    third = "d_" + "4" * 32
    locked, eviction_attempted = threading.Event(), threading.Event()
    real_lock = store._write_lock
    errors: list[BaseException] = []
    eviction_locks: list[Path] = []

    def writer() -> None:
        try:
            with real_lock(state.path):
                locked.set()
                assert eviction_attempted.wait(5)
                fresh = PlacementState.load(NETWORK, isolated_root)
                fresh.observe("third", "not_a_holder", owner_device=third, retry_after_ms=60_000)
                fresh.cool_down(
                    "openai",
                    owner_device=third,
                    credential_id=17,
                    block_scope="account",
                    until=time.time() + 3600,
                )
                store._stage_private(fresh.path, lambda handle: json.dump(fresh.to_json(), handle))
        except BaseException as exc:
            errors.append(exc)

    @contextmanager
    def observed_lock(path: Path) -> Iterator[None]:
        if path == state.path:
            eviction_locks.append(path)
            eviction_attempted.set()
        with real_lock(path):
            yield

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert locked.wait(5)
        monkeypatch.setattr(store, "_write_lock", observed_lock)
        relay.leave(record, device_id=OWNER, root=isolated_root)
    finally:
        eviction_attempted.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert eviction_locks == [state.path], "one cache lock, not nested save/re-lock"
    after = PlacementState.load(NETWORK, isolated_root).to_json()
    assert {r["owner_device"] for r in after["observations"]} == {OTHER, third}
    assert {(r["owner_device"], r["credential_id"]) for r in after["cooldowns"]} == {
        (OTHER, 7),
        (OTHER, 8),
        (third, 17),
    }
    print(
        "shared-lock writer committed first: unrelated observation/cooldown retained; "
        "departing owner removed"
    )

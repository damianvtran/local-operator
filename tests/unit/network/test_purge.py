"""``lop network uninstall``: one flag, one blast radius (R1).

The invariant this file exists for is named in the design (``mesh-network.md`` §12,
``mesh-transport-identity.md`` §16 Q1):
**``purge_identity_needs_a_named_tty_confirmation``** — the device keypair goes only
when a terminal takes a confirmation that NAMES every network the identity is in.

What the first implementation got wrong, and what each test below pins: ``--purge``
deleted only top-level files and then claimed to have deleted "this device's mesh
identity and network records", while both survived.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import identity
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store, types, wire

NETWORK = "n_0123456789abcdef01234567"


def _device(root: Path, *, name: str = "home-net") -> tuple[types.NetworkRecord, types.SecretState]:
    """A device with an identity, one network, an invite, a queue and an audit log."""
    record = types.NetworkRecord(
        network_id=NETWORK,
        name=name,
        self_device_id="",
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    minted = invite_mod.mint(record, wire.b64u(b"s" * 32), role="drive", ttl_s=600.0)
    record.invites.append(minted.record)
    store.save_invite_token(minted.record.invite_id, minted.token, root)
    state = types.SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"s" * 32))
    store.save(record, root)
    store.save_secrets(state, root)
    store.enqueue_frame("d_" + "b" * 32, {"op": "net_ping"}, removed=False, root=root)
    pending = types.PendingPairing(
        invite_id=minted.record.invite_id,
        network_id=NETWORK,
        network_name=name,
        joiner_device_id="d_" + "b" * 32,
        joiner_name="laptop",
        sas="481926",
        fingerprint="K7QM-3XPD",
        expires_at=__import__("time").time() + 60,
    )
    store.save_pending_pairing(pending, root)
    from local_operator.network.audit import AuditEvent, AuditLog

    log = AuditLog(root)
    log.record(AuditEvent(event="panic_raised", network_id=NETWORK))
    log.close()
    return record, state


def _paths(root: Path) -> dict[str, bool]:
    return {
        "record": store.record_path(NETWORK, root).exists(),
        "secrets": store.secrets_path(NETWORK, root).exists(),
        "invite": bool(list(store.outbox_dir(root).glob("*.invite"))),
        "queue": (store.outbox_dir(root) / ("d_" + "b" * 32)).is_dir(),
        "audit": store.audit_path(root).exists(),
        "pending": list(store.pending_dir(root).glob("*.pending.json")) != [],
        "identity": identity.identity_path(root).exists(),
    }


def test_purge_deletes_the_networks_and_keeps_the_device_identity(root: Path) -> None:
    """``--purge`` covers records, secrets, invites, queues, parked pairings and the
    audit log — and NOT the keypair, which every other network addresses this device
    by."""
    _device(root)
    identity.mint(root, name="this-device")
    before = _paths(root)
    assert all(before.values()), before  # nothing was missing to begin with

    result = relay.uninstall(purge=True, root=root, dry_run=True)
    # Dry run: the receipt is computed, nothing is deleted.
    assert result["deleted"]["networks"], result

    result = relay.uninstall(purge=True, root=root)
    after = _paths(root)
    assert after == {
        "record": False,
        "secrets": False,
        "invite": False,
        "queue": False,
        "audit": False,
        "pending": False,
        "identity": True,
    }, after
    assert result["identity"] == "kept"
    assert result["deleted"]["networks"] == [f"home-net ({NETWORK})"]
    assert result["deleted"]["invites"] == 1
    assert result["deleted"]["queues"] == 1
    assert result["deleted"]["pending"] == 1
    assert "audit.jsonl" in result["deleted"]["audit_files"]
    # The receipt SAYS what happened to the keypair rather than implying it went.
    assert any("NOT deleted" in step for step in result["steps"]), result["steps"]


def test_purge_receipt_is_honest_about_a_scoped_purge(root: Path) -> None:
    """The audit log is one file per install recording networks this purge did not
    cover, so a scoped purge KEEPS it and says so — deleting part of a forensic log
    is the one thing a forensic log may not undergo."""
    _device(root)
    other = types.NetworkRecord(
        network_id="n_ffffffffffffffffffffffff", name="lab", self_role="admin"
    )
    store.save(other, root)
    identity.mint(root)

    result = relay.uninstall(purge=True, networks=[NETWORK], root=root)
    assert store.record_path(NETWORK, root).exists() is False
    assert store.record_path("n_ffffffffffffffffffffffff", root).exists() is True
    assert store.audit_path(root).exists() is True
    assert result["deleted"]["audit_kept"] is True
    assert any("kept the audit log" in step for step in result["steps"]), result["steps"]


def test_purge_identity_needs_a_named_tty_confirmation(root: Path) -> None:
    """THE INVARIANT, in the design's own name.

    Three cases, and the middle one is the point: a confirmation that does not name
    every network, or that is not answered, deletes NOTHING.
    """
    _device(root)
    device = identity.mint(root, name="this-device")

    # 1. No terminal: refused outright, naming the flag that DOES work.
    with pytest.raises(types.MeshRefusal) as excinfo:
        relay.uninstall(purge_identity=True, root=root, assume_tty=False)
    assert excinfo.value.code == "purge_identity_needs_tty"
    assert "--purge" in excinfo.value.sentence
    assert identity.identity_path(root).exists(), "a refusal must not delete anything"

    # 2. A terminal, and the wrong answer: still nothing deleted.
    seen: list[str] = []

    def wrong(prompt: str) -> str:
        seen.append(prompt)
        return "yes"

    with pytest.raises(types.MeshRefusal) as excinfo:
        relay.uninstall(purge_identity=True, root=root, assume_tty=True, answer=wrong)
    assert excinfo.value.code == "purge_identity_not_confirmed"
    assert identity.identity_path(root).exists()
    # 3. The prompt NAMES the networks: that list is what the human is weighing.
    assert seen and "home-net" in seen[0] and NETWORK in seen[0]
    assert device.device_id in seen[0]

    # 4. The device id typed back: the keypair goes.
    result = relay.uninstall(
        purge=True,
        purge_identity=True,
        root=root,
        assume_tty=True,
        answer=lambda _prompt: device.device_id,
    )
    assert result["identity"] == "deleted"
    assert identity.identity_path(root).exists() is False
    assert identity.load(root) is None
    assert any("deleted the device identity" in step for step in result["steps"])


def test_purge_identity_is_refused_on_a_real_process_without_a_terminal(root: Path) -> None:
    """The production path, not the injected one: this test process has no TTY, so
    the refusal above is what an isolated run really gets."""
    identity.mint(root)
    assert relay._has_terminal() is False  # noqa: SLF001 — the guard the refusal rests on
    with pytest.raises(types.MeshRefusal) as excinfo:
        relay.uninstall(purge_identity=True, root=root)
    assert excinfo.value.code == "purge_identity_needs_tty"
    assert identity.identity_path(root).exists()


def test_uninstall_reports_an_isolated_home_instead_of_a_launchd_error(root: Path) -> None:
    """R6: a redirected HOME has no LaunchAgent, and the message must say that
    plainly — this is the shape a test harness runs in, and it must not read as a
    failure to install."""
    identity.mint(root)
    result = relay.uninstall(root=root)
    assert result["ok"] is True
    assert any("no LaunchAgent to remove here" in step for step in result["steps"]), result

    started = relay.install(port=4097)
    assert started["ok"] is False
    assert started.get("reason") == "isolated_home"
    assert "--no-start" in str(started["error"]) or "serve" in str(started["error"])
    assert "launchd" in str(started["error"])

    action = relay.service_action("start")
    assert action["ok"] is False
    assert action.get("reason") == "isolated_home"

    assert json.dumps(result, default=str)  # a receipt is JSON-serialisable for --json


def test_uninstall_never_reaches_for_launchctl_where_there_is_none(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-9: on a host with no launchd, removal is removal — not a crash.

    ``launchd.is_own_plist`` has no platform check, so on Linux the isolated-home
    guard answered "yes" and ``subprocess.run(["launchctl", ...])`` raised
    ``FileNotFoundError`` BEFORE any of the verb's work — the store stayed fully
    present, so the peers this design connects could not be cleaned up the way the
    design intends (QA round 1, F-9). ``sys.platform`` and ``shutil.which`` are
    patched to the Linux answer rather than stubbing the guard, so the test
    exercises the real ``is_supported()`` expression.
    """
    record, _state = _device(root)
    identity.mint(root)
    attempts: list[tuple[str, ...]] = []

    def _no_launchctl(*args: str) -> Any:
        attempts.append(args)
        raise AssertionError("launchctl was invoked on a host with no launchd")

    monkeypatch.setattr(relay.sys, "platform", "linux")
    monkeypatch.setattr(relay.shutil, "which", lambda _name: None)
    monkeypatch.setattr(relay, "_launchctl", _no_launchctl)

    result = relay.uninstall(purge=True, root=root)
    assert result["ok"] is True, result
    assert attempts == []
    # THE STORE REALLY WENT. This is the half QA could not reach on Linux.
    assert store.record_path(record.network_id, root).exists() is False
    assert any("no launchd on this platform" in step for step in result["steps"]), result
    # The identity is still kept: that is what `--purge` alone promises.
    assert identity.identity_path(root).exists()


def test_service_action_names_the_missing_launchd_rather_than_the_home(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The third half of one surface: an isolated HOME and a platform with no
    launchd at all are different problems, and only one of them is fixed by
    $HOME."""
    identity.mint(root)
    monkeypatch.setattr(relay.sys, "platform", "linux")
    monkeypatch.setattr(relay.shutil, "which", lambda _name: None)
    action = relay.service_action("start")
    assert action["ok"] is False
    assert action.get("reason") == "no_launchd"
    assert "serve" in str(action["error"])

"""A running relay's failed removal must never become a successful local removal.

Use the real CLI, control socket and disposable paired relays. The owner fails
in the actual ledger writer before the final wipe can be sent; its control
connection closes. Only a genuinely absent relay permits the offline spelling.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import sync
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_copy_dispatch_removal import (  # noqa: F401
    _deliver_without_ack,
    _loopback_only,
)
from tests.unit.network.test_credentials_real_link import _lop_network
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixture dependency
from tests.unit.network.test_secret_copies import (  # noqa: F401 — fixtures by import
    SECRET_NAME,
    _member_has,
    secret_mesh,
)


def test_the_loopback_guard_is_wired_before_any_case() -> None:
    with socket.socket() as sock, pytest.raises(AssertionError, match="only loopback TCP"):
        sock.connect(("192.0.2.1", 443))


def _drain_owner(mesh: Any) -> None:
    engine = sync.sync_for_relay(mesh.a)
    assert engine is not None

    def idle() -> bool:
        with engine._guard:
            return mesh.member not in engine._inflight_members

    assert net_fixtures.wait_for(idle), "the first owner exchange did not finish"


@pytest.mark.parametrize("as_json", [True, False], ids=["json", "text"])
def test_a_running_relays_failed_removal_is_not_retried_offline(
    secret_mesh: Any,  # noqa: F811 — pytest fixture imported above
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    mesh = secret_mesh
    _deliver_without_ack(mesh, monkeypatch)
    _drain_owner(mesh)
    state = sync.SyncState.load(mesh.network_id, mesh.a.root)
    # A valid pre-upgrade dispatch needs an identity persisted during planning.
    # Fail that real write, not the CLI or control_request/its result framing.
    payload = state.to_json()
    payload["copy_versions"] = {}
    store._write_private_json(state.path, payload)
    ledger_before = state.path.read_bytes()
    before_epoch = store.require_secrets(mesh.network_id, mesh.a.root).epoch
    assert store.find_own_relay(mesh.a.root) is not None
    writes: list[Path] = []
    write = store._write_private_json

    def unavailable(path: Path, document: Any) -> Path:
        if path == state.path:
            writes.append(path)
            raise OSError("synthetic removal identity persistence unavailable")
        return write(path, document)

    capsys.readouterr()
    argv = ["member", "rm", mesh.network_id, mesh.b.identity.name]
    if as_json:
        argv.append("--json")
    with monkeypatch.context() as patch:
        patch.setattr(store, "_write_private_json", unavailable)
        code = _lop_network(*argv)
    captured = capsys.readouterr()
    assert writes == [state.path], "the actual relay removal did not reach the failing writer"
    assert code != 0, (captured.out, captured.err)
    if as_json:
        refusal = json.loads(captured.out)
        assert refusal["ok"] is False
        assert refusal["code"] == "relay_control_failed"
        message = refusal["message"]
    else:
        message = captured.out + captured.err
    assert "did not return a result" in message
    assert "no offline removal was attempted" in message
    assert "may be incomplete" in message
    assert "applied locally" not in captured.out + captured.err
    record = store.load(mesh.network_id, mesh.a.root)
    assert mesh.member in [m.device_id for m in record.active_members()]
    assert store.require_secrets(mesh.network_id, mesh.a.root).epoch == before_epoch
    assert state.path.read_bytes() == ledger_before
    assert _member_has(mesh.b.root, SECRET_NAME)
    assert store.find_own_relay(mesh.a.root) is not None
    # Restore the writer and retry through the same still-running relay: its
    # normal bound wipe/removal remains the only successful online path.
    assert _lop_network(*argv) == 0
    successful = capsys.readouterr()
    if as_json:
        receipt = json.loads(successful.out)
        assert receipt["copies"] == receipt["wiped"] == 1
        assert receipt["unacknowledged"] == 0
    assert not _member_has(mesh.b.root, SECRET_NAME)


@pytest.mark.parametrize("as_json", [True, False], ids=["json", "text"])
def test_a_truly_absent_relay_still_allows_an_explicit_offline_removal(
    secret_mesh: Any,  # noqa: F811 — pytest fixture imported above
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    mesh = secret_mesh
    _deliver_without_ack(mesh, monkeypatch)
    _drain_owner(mesh)
    mesh.a.stop()
    assert store.find_own_relay(mesh.a.root) is None
    before_epoch = store.require_secrets(mesh.network_id, mesh.a.root).epoch
    capsys.readouterr()
    argv = ["member", "rm", mesh.network_id, mesh.b.identity.name]
    if as_json:
        argv.append("--json")
    assert _lop_network(*argv) == 0
    captured = capsys.readouterr()
    if as_json:
        receipt = json.loads(captured.out)
        assert receipt["ok"] is True
        assert receipt["copies"] == receipt["unacknowledged"] == 1
        assert receipt["wiped"] == 0
        assert receipt["relay"] == "not running — applied locally and queued"
    else:
        assert "relay is not running" in captured.out
        assert "never acknowledged" in captured.out
    record = store.load(mesh.network_id, mesh.a.root)
    assert mesh.member not in [m.device_id for m in record.active_members()]
    assert store.require_secrets(mesh.network_id, mesh.a.root).epoch == before_epoch + 1
    assert _member_has(mesh.b.root, SECRET_NAME)

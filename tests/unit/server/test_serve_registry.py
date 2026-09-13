"""The ``serve`` daemon's rendezvous record.

The record exists so a process can answer two questions another process asks
about a running ``lop serve``: WHERE is it listening, and WHICH install is it.
These tests pin the parts of that contract a reader depends on — the schema
(additive, so a record written by another build still parses), the permissions
and atomicity inherited from the shared publication path, the liveness rule
shared with the session namespace rather than re-implemented, and the announced
address, which is what makes ``--port 0`` recordable at all.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from local_operator.server import registry as serve_registry
from local_operator.session.runtime import registry as session_registry
from local_operator.session.runtime.types import (
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_TIMEOUT_S,
    RUN_DIRNAME,
    SERVE_RUN_DIRNAME,
    SessionRecord,
)


def make_record(**overrides: object) -> serve_registry.ServeRecord:
    fields: dict[str, object] = {
        "pid": os.getpid(),
        "host": "127.0.0.1",
        "port": 54321,
        "instance_id": "instance-under-test",
        "version": "0.54.32",
        "source_ref": "abc1234",
        "prefix": "/tmp/venvs/serve",
        "install_kind": "uv-tool",
        "desktop": False,
    }
    fields.update(overrides)
    return serve_registry.ServeRecord(**fields)  # type: ignore[arg-type]


def test_the_record_lives_in_its_own_namespace(tmp_path: Path) -> None:
    """``run/serve``, not ``run/mobile``: a session reader must never see this.

    Asserted by path rather than by scanning, so a future change that put the
    file back among the sessions fails here with the path that did it.
    """
    assert SERVE_RUN_DIRNAME == "run/serve"
    assert RUN_DIRNAME != SERVE_RUN_DIRNAME
    record = make_record()
    path = serve_registry.publish(record, root=tmp_path)
    assert path == tmp_path / SERVE_RUN_DIRNAME / f"{record.pid}.json"
    assert serve_registry.record_path(record.pid, tmp_path) == path
    # A session-namespace reader sees nothing: no phantom session row.
    assert session_registry.scan(tmp_path) == []


def test_publish_is_atomic_and_0600_under_a_0700_directory(tmp_path: Path) -> None:
    """The permissions ARE the authorization model (the shared implementation's
    rule, inherited here rather than re-decided)."""
    path = serve_registry.publish(make_record(), root=tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    # No staging files left behind: the write is a rename, not a copy.
    assert [p.name for p in path.parent.iterdir()] == [f"{path.stem}.json"]
    # And the payload is the record, whole.
    data = json.loads(path.read_text())
    assert data["instance_id"] == "instance-under-test"
    assert set(data) == {
        "pid",
        "host",
        "port",
        "instance_id",
        "version",
        "source_ref",
        "prefix",
        "install_kind",
        "desktop",
        "claim_key",
        "started_at",
        "heartbeat_at",
    }


def test_unpublish_is_best_effort_and_namespace_scoped(tmp_path: Path, monkeypatch) -> None:
    """An exit path must never raise over a missing file, and must not delete a
    session's record on the way out (same pid, different namespace)."""
    session_like = tmp_path / RUN_DIRNAME
    session_like.mkdir(parents=True)
    peer = session_like / f"{os.getpid()}.json"
    peer.write_text("{}")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    path = serve_registry.publish(make_record())
    serve_registry.unpublish(os.getpid())
    serve_registry.unpublish(os.getpid())  # twice: no raise
    assert not path.exists()
    assert peer.exists(), "the session namespace is a different namespace"


def test_serve_record_round_trips_and_ignores_unknown_keys() -> None:
    record = make_record()
    data = record.to_json()
    data["a_field_from_the_future"] = {"nested": True}
    restored = serve_registry.ServeRecord.from_json(data)
    assert restored == record
    assert not hasattr(restored, "a_field_from_the_future")


def test_absent_keys_fall_back_to_their_defaults() -> None:
    """The additive half of the ``from_json`` contract.

    A record written by a binary that predates a defaulted field parses here
    with the default, and a record written by a NEWER one that added another
    field parses too (the unknown-key case above). Neither may raise: a raise
    is what ``scan`` treats as a torn file and REAPS, so a strict reader would
    delete a live daemon's record during an upgrade window.
    """
    payload = make_record().to_json()
    for key in ("claim_key", "started_at", "heartbeat_at"):
        del payload[key]

    restored = serve_registry.ServeRecord.from_json(payload)

    assert restored.claim_key == ""
    assert restored.started_at > 0
    assert restored.heartbeat_at > 0


def test_a_record_missing_an_identity_field_is_not_this_shape() -> None:
    """The boundary of that contract: only the defaulted tail may be absent.

    ``pid``/``port``/``instance_id`` and the install identity have no sensible
    default — a record that lacks one is not a serve record at all — and this
    pins where that line is, so adding a field is a deliberate choice between
    "defaulted tail" and "required identity".
    """
    payload = make_record().to_json()
    del payload["instance_id"]
    with pytest.raises(TypeError):
        serve_registry.ServeRecord.from_json(payload)


def test_scan_classifies_live_wedged_and_stale_against_the_shared_rule(tmp_path: Path) -> None:
    """The liveness rule is the SESSION registry's, asked for this namespace.

    A copy of the rule is the failure mode this asserts against: two
    implementations would be free to disagree, and the reader that got the
    other answer is the one nobody was looking at. So the same three shapes are
    published into BOTH namespaces and the states are compared pairwise.
    """

    def session_peer(pid: int) -> SessionRecord:
        return SessionRecord(
            pid=pid,
            kind="tui",
            session_id="s",
            conversation_name="demo",
            cwd="/tmp",
            model_label="m",
            control_port=1,
            control_key="k" * 64,
        )

    live = make_record()
    dead = make_record(pid=2**22 - 3)  # a pid that does not exist
    wedged = make_record()

    serve_registry.publish(live, root=tmp_path)
    dead_path = serve_registry.publish(dead, root=tmp_path)
    # Written directly rather than through publish(), which stamps a fresh
    # heartbeat by design — a wedged record is one whose heartbeat stopped.
    wedged.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 1
    (serve_registry.record_path(wedged.pid, tmp_path)).write_text(json.dumps(wedged.to_json()))

    session_registry.publish(session_peer(live.pid), root=tmp_path)
    session_registry.publish(session_peer(dead.pid), root=tmp_path)
    stale_peer = session_peer(wedged.pid)
    stale_peer.heartbeat_at = wedged.heartbeat_at
    session_registry.record_path(wedged.pid, tmp_path).write_text(json.dumps(stale_peer.to_json()))

    # ``ServeRecord`` is an ordinary dataclass and therefore unhashable, so the
    # states are keyed by pid — which is the key both scans use anyway.
    serve_states = {r.pid: state for r, state in serve_registry.scan(tmp_path)}
    session_states = {r.pid: state for r, state in session_registry.scan(tmp_path)}

    assert serve_states == {
        live.pid: "live",
        dead.pid: "stale",
        wedged.pid: "wedged",
    }
    assert serve_states == session_states, "one rule, two namespaces"
    assert not dead_path.exists(), "stale means the reader reaped the file"


def test_heartbeat_loop_rewrites_the_record(tmp_path: Path, monkeypatch) -> None:
    """The heartbeat is what makes a live daemon distinguishable from a wedged
    one, so it is a write of the whole record, on a timer."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    record = make_record()
    publisher = serve_registry.publisher(record)
    path = serve_registry.record_path(record.pid, tmp_path)
    first = json.loads(path.read_text())["heartbeat_at"]

    # Patched to something test-scale; the production interval is asserted
    # separately below, so this cannot hide a drift in the constant.
    monkeypatch.setattr(serve_registry, "HEARTBEAT_INTERVAL_S", 0.05)

    async def drive() -> None:
        task = asyncio.create_task(serve_registry.heartbeat_loop(publisher))
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(drive())

    assert json.loads(path.read_text())["heartbeat_at"] > first
    publisher.close()


def test_the_heartbeat_interval_is_the_shared_constant() -> None:
    """15 s, from ``types``: one freshness budget for both record kinds, so a
    reader needs a single rule for "is this alive"."""
    assert serve_registry.HEARTBEAT_INTERVAL_S == HEARTBEAT_INTERVAL_S == 15.0
    assert HEARTBEAT_TIMEOUT_S == 45.0


def test_advertised_address_reports_what_was_announced(monkeypatch) -> None:
    monkeypatch.delenv(serve_registry.SERVE_HOST_ENV, raising=False)
    monkeypatch.delenv(serve_registry.SERVE_PORT_ENV, raising=False)
    assert serve_registry.advertised_address() == ("", 0)

    serve_registry.announce_address("127.0.0.1", 58474)
    assert serve_registry.advertised_address() == ("127.0.0.1", 58474)

    # A garbled port is "unknown", never a guess: a record naming the wrong
    # port is the failure this module exists to remove.
    monkeypatch.setenv(serve_registry.SERVE_PORT_ENV, "not-a-port")
    assert serve_registry.advertised_address() == ("127.0.0.1", 0)


def test_build_record_reads_this_processs_identity(monkeypatch) -> None:
    from local_operator.update import install_kind, installed_build

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "desktop-token")
    serve_registry.announce_address("127.0.0.1", 58474)

    record = serve_registry.build_record(instance_id="minted")

    assert record.pid == os.getpid()
    assert (record.host, record.port) == ("127.0.0.1", 58474)
    assert record.instance_id == "minted"
    assert record.prefix == sys.prefix
    assert record.install_kind == install_kind().value
    assert (record.version, record.source_ref) == (
        installed_build().version,
        installed_build().source_ref,
    )
    assert record.desktop is True, "the desktop plane's own predicate is the env"
    assert record.claim_key == "", "reserved for the claim handshake, empty until then"


def test_build_record_reports_the_desktop_plane_as_unset(monkeypatch) -> None:
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
    assert serve_registry.build_record(instance_id="x").desktop is False

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
from types import SimpleNamespace

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

    The three cases MUST carry three DIFFERENT pids, and that is load-bearing
    rather than tidy: the records share one namespace keyed by pid, so two cases
    on one pid mean the later write replaces the earlier file, and the
    expectation for the earlier one is then asserting a record that no longer
    exists — an assertion that cannot fail. (This test did exactly that: `live`
    and `wedged` both defaulted to this process's pid, so nothing was ever
    classified `live` here.)
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
    # A pid that is alive and is NOT this process: the parent is the one live
    # pid a test can name without forking one, and `wedged` must differ from
    # `live` for the reason in the docstring. Asserted so the premise is stated
    # rather than assumed — a dead pid here would classify `stale` and fail
    # below with the reason already named.
    wedged = make_record(pid=os.getppid())
    assert session_registry.pid_alive(wedged.pid), "the wedged case needs a LIVE pid"

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
    # states are keyed by pid — which is the key both scans use anyway. Each
    # case is then asserted on ITS OWN pid, so a collapsed dict cannot pass for
    # a matched one.
    serve_states = {r.pid: state for r, state in serve_registry.scan(tmp_path)}
    session_states = {r.pid: state for r, state in session_registry.scan(tmp_path)}

    assert serve_states.get(live.pid) == "live"
    assert serve_states.get(dead.pid) == "stale"
    assert serve_states.get(wedged.pid) == "wedged"
    assert len(serve_states) == 3, "three cases, three records: no pid is shared"
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


def fake_app() -> SimpleNamespace:
    """The one attribute the announce/read pair touches: ``app.state``.

    Real enough for this contract: Starlette's ``State`` maps
    ``getattr``/``setattr`` onto a dict, which is what a namespace is, and this
    module is stdlib-only by contract so its tests do not import Starlette to
    stand in for a dict.
    """
    return SimpleNamespace(state=SimpleNamespace())


def test_advertised_address_is_what_was_announced_to_this_process(monkeypatch) -> None:
    """The in-process channel, and ``None`` when nothing announced one.

    ``None`` rather than a ``("", 0)`` placeholder because the caller uses it to
    decide whether to publish at all (see the lifespan test): a boot that cannot
    name an address is not a daemon anyone can dial.
    """
    monkeypatch.delenv(serve_registry.SERVE_ANNOUNCE_ENV, raising=False)
    app = fake_app()
    assert serve_registry.advertised_address(app) is None

    serve_registry.announce_address(app, "127.0.0.1", 58474)
    assert serve_registry.advertised_address(app) == ("127.0.0.1", 58474)


def test_an_announcement_addressed_to_somebody_else_is_ignored_and_cleared(monkeypatch) -> None:
    """The false-rendezvous leak, at the registry: an announce that is not ours.

    ``--reload`` is the only path that uses the environment, and its value names
    the process that made it. A process that merely INHERITED the variable — an
    agent shell inside a daemon, a wrapper, a nested boot of the same app — is
    not that process's child, so it takes nothing from it. Consumed either way,
    so nothing this process goes on to spawn can find it and re-publish its
    ancestor's listener as its own.

    The positive half of this gate (the announcement the ``--reload`` child
    DOES take) needs a real ``multiprocessing`` spawn of uvicorn's child, which
    no test here does; it is exercised by the CLI test that asserts what
    ``serve_command`` writes, plus the parent-pid rule asserted by construction.
    """
    # Our own pid: alive, but not our multiprocessing spawner — a pytest process
    # has no multiprocessing parent at all, which is exactly the shape refused.
    monkeypatch.setenv(serve_registry.SERVE_ANNOUNCE_ENV, f"{os.getpid()} 10.0.0.1 9000")
    assert serve_registry.advertised_address(fake_app()) is None
    assert serve_registry.SERVE_ANNOUNCE_ENV not in os.environ, "read-and-cleared"

    # Garbled values are consumed and never guessed at: a record naming the
    # wrong address is the failure this module exists to remove.
    monkeypatch.setenv(serve_registry.SERVE_ANNOUNCE_ENV, "not-an-announcement")
    assert serve_registry.advertised_address(fake_app()) is None
    assert serve_registry.SERVE_ANNOUNCE_ENV not in os.environ


def test_the_in_process_announce_outranks_and_clears_an_inherited_one(monkeypatch) -> None:
    """A daemon that inherited a stranger's announcement announces its own.

    The explicit announce wins, and the inherited value is consumed with it, so
    a later reader in this process cannot reach a stale address from an
    ancestor.
    """
    monkeypatch.setenv(serve_registry.SERVE_ANNOUNCE_ENV, f"{os.getpid()} 10.0.0.1 9000")
    app = fake_app()
    serve_registry.announce_address(app, "127.0.0.1", 58474)

    assert serve_registry.advertised_address(app) == ("127.0.0.1", 58474)
    assert serve_registry.SERVE_ANNOUNCE_ENV not in os.environ


def test_build_record_reads_this_processs_identity(monkeypatch) -> None:
    from local_operator.update import install_kind, installed_build

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "desktop-token")

    record = serve_registry.build_record(instance_id="minted", announced=("127.0.0.1", 58474))

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


def test_a_wildcard_bind_is_recorded_as_the_loopback_it_is_dialable_on() -> None:
    """The record is read by another process to DIAL it, so it carries a
    dialable host: a wildcard bind means "every interface", which is not an
    address, and it is reachable on its own family's loopback."""
    assert (
        serve_registry.build_record(instance_id="x", announced=("0.0.0.0", 1111)).host
        == "127.0.0.1"
    )
    assert serve_registry.build_record(instance_id="x", announced=("::", 1111)).host == "::1"
    # An explicit address is recorded verbatim: rewriting it would be guessing at
    # a route (and the dialer brackets an IPv6 literal when it builds a URL).
    assert (
        serve_registry.build_record(instance_id="x", announced=("127.0.0.1", 1111)).host
        == "127.0.0.1"
    )


def test_build_record_reports_the_desktop_plane_as_unset(monkeypatch) -> None:
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN", raising=False)
    assert serve_registry.build_record(instance_id="x", announced=("127.0.0.1", 1)).desktop is False


def test_a_daemon_nobody_owns_publishes_a_claim_key(tmp_path: Path, monkeypatch) -> None:
    """The other half of the handshake: the key exists exactly where the record
    exists, and nowhere else a reader could pick it up.

    ``build_record`` mints it rather than the HTTP app, because a UI that
    discovers a daemon in the same instant the record appears must find a key
    already there — a record published without one would be a daemon the app
    can see and still not attach to, which is the failure this change removes.
    """
    from local_operator.server import desktop

    monkeypatch.delenv(desktop.TOKEN_ENV, raising=False)
    monkeypatch.setattr(desktop, "_CLAIMED", None)

    record = serve_registry.build_record(instance_id="minted", announced=("127.0.0.1", 58474))
    assert record.desktop is False
    # 32 random bytes, base64url-encoded, unpadded.
    assert len(record.claim_key) == 43
    assert serve_registry.build_record(
        instance_id="minted-again", announced=("127.0.0.1", 58474)
    ).claim_key not in {"", record.claim_key}

    path = serve_registry.publish(record, root=tmp_path)
    assert json.loads(path.read_text())["claim_key"] == record.claim_key
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_daemon_the_app_started_publishes_no_claim_key(monkeypatch) -> None:
    """Nothing to claim: the environment already governs this plane, and a key
    would invite a second principal to take a daemon the app is driving."""
    from local_operator.server import desktop

    monkeypatch.setenv(desktop.TOKEN_ENV, "desktop-token")
    monkeypatch.setattr(desktop, "_CLAIMED", None)

    record = serve_registry.build_record(instance_id="appowned", announced=("127.0.0.1", 1234))
    assert record.desktop is True
    assert record.claim_key == ""


def test_the_claim_key_is_never_logged(monkeypatch, caplog) -> None:
    """The record is the key's only channel, so no log line may carry it."""
    import logging

    from local_operator.server import desktop

    monkeypatch.delenv(desktop.TOKEN_ENV, raising=False)
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    probe = "log-capture-probe"

    with caplog.at_level(logging.DEBUG):
        logging.getLogger("local_operator.server").debug("capture %s", probe)
        record = serve_registry.build_record(instance_id="minted", announced=("127.0.0.1", 1))

    messages = "\n".join(r.getMessage() for r in caplog.records) + caplog.text
    assert probe in messages, "log capture is not working; the assertion below is vacuous"
    assert record.claim_key not in messages

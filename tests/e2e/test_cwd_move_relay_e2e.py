"""Recorded cwd moves survive the real peer-control → mesh engage producer.

The relay used to turn omission into HOME before the spawn resolver. A socket
ack was still successful while the actual child cancelled the move at boot, so
this regression must read both the published cwd and the durable terminal move.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import projection, relay, store
from local_operator.session import cwd_move
from local_operator.session.runtime import launch, registry
from tests.e2e.harness import NO_NOTIFY_ENV
from tests.e2e.test_cwd_move_e2e import _reap, _seed
from tests.unit.network.conftest import root  # noqa: F401
from tests.unit.network.test_relay_e2e import devices  # noqa: F401
from tests.unit.network.test_session_plane import (  # noqa: F401
    _dial_to,
    _listen,
    _pair,
    peer_pair,
)

# root/devices/peer_pair are pytest fixture imports, not unused runtime helpers.

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("choice", ["omitted", "home", "third"])
def test_remote_engage_preserves_cwd_intent_through_the_real_mesh(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    pair = request.getfixturevalue("peer_pair")
    server_a, server_b, _, _ = pair
    record, _, _ = _pair(pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _dial_to(server_a, record, host_b, port_b)
    sid = "cafe3275ee05"
    _seed(server_b.root, sid)
    source, destination, third = (tmp_path / name for name in ("source", "destination", "third"))
    for path in (source, destination, third):
        path.mkdir()
    intent = cwd_move.enqueue_self_move(
        server_b.root, sid, requested_cwd=str(destination), effective_cwd=str(source), owner_pid=1
    )
    assert cwd_move.accept_handoff(server_b.root, sid, intent) is not None
    # Two synthetic devices share a process in this fixture; bind its child
    # environment to the OWNER, as a real owner's separate relay process would.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    for key, value in NO_NOTIFY_ENV.items():
        monkeypatch.setenv(key, value)
    children: list[subprocess.Popen[bytes]] = []
    real_spawn = launch._spawn_runtime

    def observe(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = real_spawn(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(launch, "_spawn_runtime", observe)
    fields = {} if choice == "omitted" else {"cwd": str(Path.home() if choice == "home" else third)}
    expected = str(destination.resolve()) if choice == "omitted" else fields["cwd"]
    try:
        first = None
        for attempt in range(2):
            if attempt == 0:
                owner = projection.RemoteOwner(
                    config_dir=server_a.root,
                    root=server_a.root,
                    session_id=sid,
                    facts=projection.RemoteSessionFacts(
                        session_id=sid,
                        device_id=server_b.identity.device_id,
                        device_name=server_b.identity.name,
                    ),
                )
                asyncio.run(owner.engage(cwd=fields.get("cwd", ""), warm=launch.WarmErrand()))
                detail = {"remote_owner": "engage returned successfully"}
            else:
                local_record = store.find_own_relay(server_a.root)
                assert local_record is not None
                # The real RemoteOwner uses this spawn-sized budget, not the
                # generic test helper's 5-second control-socket default.
                reply = relay.control_request(
                    local_record,
                    "peer_session_engage",
                    timeout=relay.engage_client_bound_s(),
                    peer=server_b.identity.device_id,
                    session_id=sid,
                    # Omit cwd on the second cold start: an explicit override
                    # must have cancelled the old move exactly once, not merely
                    # hidden it until the next caller has no directory opinion.
                )
                assert (
                    reply is not None and reply["op"] == "ack" and reply["detail"]["engaged"]
                ), reply
                detail = reply["detail"]
            records = [r for r, _ in registry.scan(server_b.root) if r.session_id == sid]
            settled = cwd_move.read_move_intent(server_b.root, sid)
            wanted = expected if attempt == 0 or choice == "omitted" else str(Path.home())
            assert records and records[0].cwd == wanted, records
            assert settled is not None
            assert settled.stage == ("committed" if choice == "omitted" else "superseded")
            print(
                choice,
                attempt + 1,
                detail,
                "cwd",
                records[0].cwd,
                "stage",
                settled.stage,
                flush=True,
            )
            if first is not None:
                assert settled == first, "terminal move must not be superseded a second time"
            first = settled
            assert len(children) == attempt + 1, "each attempt really started a cold native owner"
            _reap(children[-1])
    finally:
        for child in children:
            _reap(child)
